"""Sample independent detector/LED count-space time-shift and scale posteriors."""

import argparse
import csv
import json
from pathlib import Path

import emcee
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fit_wfi_tmm_numpyro import apply_fixed_corrections, load_tile_statistics


RESET_DAY = 15.064405023148149
SEED = 20260930
WARMUP_STEPS = 2500
PRODUCTION_STEPS = 6000
NWALKERS = 128
LED_VALUES = tuple(range(1, 7))


def _detector_from_path(path):
    return path.name.split("_tile_log_statistics_", 1)[0].upper()


def _detector_sort_key(value):
    value = str(value).upper()
    suffix = value[3:] if value.startswith("WFI") else value
    return (0, int(suffix)) if suffix.isdigit() else (1, value)


def _statistics_paths(path, detectors=None):
    path = Path(path)
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(path.glob("wfi*_tile_log_statistics_512.csv"))
    if not paths:
        raise ValueError(f"No raw 512-pixel tile statistics found at {path}")
    by_detector = {_detector_from_path(item): item for item in paths}
    if detectors is None:
        selected = sorted(by_detector, key=_detector_sort_key)
    else:
        selected = [str(value).strip().upper() for value in detectors]
        if len(set(selected)) != len(selected):
            raise ValueError("Detector list contains duplicates")
    missing = [detector for detector in selected if detector not in by_detector]
    if missing:
        raise ValueError(f"Statistics are missing requested detectors: {missing}")
    return [(detector, by_detector[detector]) for detector in selected]


def _correction_path(path, detector, n_detectors):
    path = Path(path)
    if path.is_file():
        return path
    candidates = (
        path / f"{detector.lower()}_count_space_fixed_corrections_three_events.csv",
        path / f"{detector.lower()}_count_space_fixed_corrections.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if n_detectors == 1:
        raise ValueError(f"No count-space correction table found for {detector} in {path}")
    raise ValueError(
        f"No detector-specific count-space correction table found for {detector} in {path}"
    )


def _load_detector_model(
    detector,
    statistics_path,
    corrections_path,
    reset_day,
    additional_acm_step_repeat,
    pre_reset_acm_step_repeat,
):
    data = load_tile_statistics(
        statistics_path,
        additional_acm_step_repeat=additional_acm_step_repeat,
        pre_reset_acm_step_repeat=pre_reset_acm_step_repeat,
    )
    corrected = apply_fixed_corrections(data, corrections_path)
    if corrected.get("fixed_correction_space") != "counts":
        raise ValueError(
            f"{detector} correction table must be marked as count-space: {corrections_path}"
        )
    counts = np.asarray(data["counts"], dtype=float)
    values = np.asarray(corrected["sum_current_counts"], dtype=float) / counts
    return {
        "detector": detector,
        "data": data,
        "values": values,
        "counts": counts,
        "corrections_path": str(Path(corrections_path).expanduser().resolve()),
        "reset_day": float(reset_day),
    }


def load_model_data(
    statistics_path,
    corrections_path,
    reset_day,
    additional_acm_step_repeat,
    pre_reset_acm_step_repeat,
    detectors=None,
):
    statistics = _statistics_paths(statistics_path, detectors)
    detector_models = []
    for detector, path in statistics:
        correction_path = _correction_path(path=corrections_path, detector=detector, n_detectors=len(statistics))
        detector_models.append(
            _load_detector_model(
                detector,
                path,
                correction_path,
                reset_day,
                additional_acm_step_repeat,
                pre_reset_acm_step_repeat,
            )
        )

    segments = []
    for detector_index, detector_model in enumerate(detector_models):
        data = detector_model["data"]
        values = detector_model["values"]
        counts = detector_model["counts"]
        days = np.asarray(data["days"], dtype=float)
        leds = np.asarray(data["led"], dtype=int)
        pre_mask = days < reset_day
        post_mask = ~pre_mask
        for led in LED_VALUES:
            pre_indices = np.flatnonzero(pre_mask & (leds == led))
            post_indices = np.flatnonzero(post_mask & (leds == led))
            if not pre_indices.size or not post_indices.size:
                raise ValueError(f"{detector_model['detector']} LED {led} lacks pre/post data")
            pre_days = days[pre_indices]
            post_days = days[post_indices]
            shift_lower = float(pre_days[0] - np.min(post_days))
            shift_upper = float(pre_days[-1] - np.max(post_days))
            if not shift_lower < shift_upper:
                raise ValueError(
                    f"{detector_model['detector']} LED {led} has no full-overlap shift prior"
                )
            segments.append(
                {
                    "detector": detector_model["detector"],
                    "detector_index": detector_index,
                    "led": led,
                    "reset_day": float(reset_day),
                    "parameter_index": len(segments),
                    "pre_days": pre_days,
                    "post_days": post_days,
                    "pre_values": values[:, pre_indices],
                    "post_values": values[:, post_indices],
                    "post_weights": counts[:, post_indices],
                    "shift_prior_lower": shift_lower,
                    "shift_prior_upper": shift_upper,
                }
            )

    mean_pair_count = float(
        np.mean(np.concatenate([segment["post_weights"].ravel() for segment in segments]))
    )
    for segment in segments:
        segment["post_weights"] = segment["post_weights"] / mean_pair_count
    return {
        "detectors": [model["detector"] for model in detector_models],
        "detector_models": detector_models,
        "segments": segments,
        "reset_day": float(reset_day),
        "n_detectors": len(detector_models),
        "n_tiles": int(detector_models[0]["data"]["n_tiles"]),
        "n_segments": len(segments),
        "n_data": float(sum(np.sum(segment["post_weights"]) for segment in segments)),
        "mean_pair_count": mean_pair_count,
        "correction_paths": {
            model["detector"]: model["corrections_path"] for model in detector_models
        },
    }


def _segment_prediction(segment, delta_t_from_t0):
    delta_t = np.asarray(delta_t_from_t0, dtype=float)
    scalar = delta_t.ndim == 0
    delta_t = np.atleast_1d(delta_t)
    shifted_days = segment["post_days"][None, :] + (
        delta_t[:, None] - segment["reset_day"]
    )
    upper = np.searchsorted(segment["pre_days"], shifted_days, side="right")
    lower = np.clip(upper - 1, 0, len(segment["pre_days"]) - 2)
    upper = lower + 1
    fraction = (shifted_days - segment["pre_days"][lower]) / (
        segment["pre_days"][upper] - segment["pre_days"][lower]
    )
    lower_values = np.transpose(segment["pre_values"][:, lower], (1, 0, 2))
    upper_values = np.transpose(segment["pre_values"][:, upper], (1, 0, 2))
    prediction = lower_values * (1.0 - fraction[:, None, :]) + upper_values * fraction[:, None, :]
    return prediction[0] if scalar else prediction


def interpolated_predictions(model, delta_t_from_t0):
    """Return one interpolated prediction/observation tuple per detector/LED."""
    delta_t = np.asarray(delta_t_from_t0, dtype=float).reshape(-1)
    if delta_t.size != model["n_segments"]:
        raise ValueError(
            f"Expected {model['n_segments']} detector/LED shift values, got {delta_t.size}"
        )
    return [
        (
            _segment_prediction(segment, delta_t[index]),
            segment["post_values"],
            segment["post_weights"],
        )
        for index, segment in enumerate(model["segments"])
    ]


def _parameter_arrays(model, parameters):
    parameters = np.asarray(parameters, dtype=float)
    scalar = parameters.ndim == 1
    parameters = np.atleast_2d(parameters)
    n_segments = model["n_segments"]
    if parameters.shape[1] != 2 * n_segments + 1:
        raise ValueError(
            f"Expected {2 * n_segments + 1} parameters, got {parameters.shape[1]}"
        )
    return scalar, parameters, parameters[:, :n_segments], parameters[:, n_segments : 2 * n_segments], parameters[:, -1]


def log_probability(model, parameters):
    scalar, parameters, delta_t, log_grey, log_sigma = _parameter_arrays(model, parameters)
    lower = np.asarray([segment["reset_day"] + segment["shift_prior_lower"] for segment in model["segments"]])
    upper = np.asarray([segment["reset_day"] + segment["shift_prior_upper"] for segment in model["segments"]])
    valid = (
        np.isfinite(parameters).all(axis=1)
        & np.all((delta_t >= lower[None, :]) & (delta_t <= upper[None, :]), axis=1)
        & np.all((log_grey > -2.0) & (log_grey < 2.0), axis=1)
        & (log_sigma > -5.0)
        & (log_sigma < 8.0)
    )
    result = np.full(len(parameters), -np.inf, dtype=float)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        weighted_sse = np.zeros(valid_indices.size, dtype=float)
        valid_delta_t = delta_t[valid_indices]
        valid_log_grey = log_grey[valid_indices]
        for segment_index, segment in enumerate(model["segments"]):
            prediction = _segment_prediction(
                segment, valid_delta_t[:, segment_index]
            )
            residual = segment["post_values"][None, :, :] - np.exp(
                valid_log_grey[:, segment_index, None, None]
            ) * prediction
            weighted_sse += np.sum(
                segment["post_weights"][None, :, :] * residual**2, axis=(1, 2)
            )
        sigma = np.exp(log_sigma[valid_indices])
        result[valid_indices] = -0.5 * (
            weighted_sse / sigma**2
            + model["n_data"] * np.log(2.0 * np.pi * sigma**2)
        )
    return float(result[0]) if scalar else result


def segment_log_probability(segment, parameters):
    """Evaluate the independent three-parameter posterior for one detector/LED."""
    parameters = np.asarray(parameters, dtype=float)
    scalar = parameters.ndim == 1
    parameters = np.atleast_2d(parameters)
    delta_t = parameters[:, 0]
    log_grey = parameters[:, 1]
    log_sigma = parameters[:, 2]
    lower = segment["reset_day"] + segment["shift_prior_lower"]
    upper = segment["reset_day"] + segment["shift_prior_upper"]
    valid = (
        np.isfinite(parameters).all(axis=1)
        & (delta_t >= lower)
        & (delta_t <= upper)
        & (log_grey > -2.0)
        & (log_grey < 2.0)
        & (log_sigma > -5.0)
        & (log_sigma < 8.0)
    )
    result = np.full(len(parameters), -np.inf, dtype=float)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        prediction = _segment_prediction(segment, delta_t[valid_indices])
        residual = segment["post_values"][None, :, :] - np.exp(
            log_grey[valid_indices, None, None]
        ) * prediction
        weighted_sse = np.sum(
            segment["post_weights"][None, :, :] * residual**2,
            axis=(1, 2),
        )
        sigma = np.exp(log_sigma[valid_indices])
        n_data = float(np.sum(segment["post_weights"]))
        result[valid_indices] = -0.5 * (
            weighted_sse / sigma**2 + n_data * np.log(2.0 * np.pi * sigma**2)
        )
    return float(result[0]) if scalar else result


def _quantiles(values):
    return {
        str(level): float(np.quantile(values, level))
        for level in (0.025, 0.16, 0.5, 0.84, 0.975)
    }


def _posterior_parameter_arrays(model, chain):
    flat = np.asarray(chain, dtype=float).reshape(-1, chain.shape[-1])
    n_segments = model["n_segments"]
    return flat[:, :n_segments], np.exp(flat[:, n_segments : 2 * n_segments]), np.exp(flat[:, -1])


def _independent_summary(model, results, initial_delta_t=None, initial_scale=None):
    parameters = {}
    rows = []
    for segment, result in zip(model["segments"], results):
        chain = result["chain"]
        flat = chain.reshape(-1, 3)
        delta_values = flat[:, 0]
        grey_values = np.exp(flat[:, 1])
        sigma_values = np.exp(flat[:, 2])
        delta_quantiles = _quantiles(delta_values)
        grey_quantiles = _quantiles(grey_values)
        rows.append(
            {
                "detector": segment["detector"],
                "led": segment["led"],
                "delta_t_from_t0_days": delta_quantiles,
                "shift_days": _quantiles(delta_values - model["reset_day"]),
                "multiplicative_scale": grey_quantiles,
                "scale_change_percent": _quantiles(100.0 * (grey_values - 1.0)),
                "shift_prior_days": [segment["shift_prior_lower"], segment["shift_prior_upper"]],
                "sigma_counts": _quantiles(sigma_values),
                "acceptance_fraction_mean": float(np.mean(result["acceptance_fraction"])),
                "autocorrelation_time": [float(value) for value in result["autocorrelation_time"]],
            }
        )
        parameters[f"{segment['detector']}_LED{segment['led']}"] = rows[-1]
    return {
        "parameterization": "independent delta_t_from_t0, multiplicative scale, and sigma_counts per detector/LED",
        "reset_day": model["reset_day"],
        "detectors": model["detectors"],
        "n_tiles_per_detector": model["n_tiles"],
        "n_data": model["n_data"],
        "correction_paths": model["correction_paths"],
        "initial_delta_t_from_t0_days": initial_delta_t,
        "initial_scale": None if initial_scale is None else float(initial_scale),
        "posterior_quantiles": {
            "parameters": parameters,
        },
    }


def _write_parameter_csv(path, model, results):
    fields = (
        "detector",
        "led",
        "delta_t_p025",
        "delta_t_p16",
        "delta_t_median",
        "delta_t_p84",
        "delta_t_p975",
        "shift_p025",
        "shift_p16",
        "shift_median",
        "shift_p84",
        "shift_p975",
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
        for segment, result in zip(model["segments"], results):
            flat = result["chain"].reshape(-1, 3)
            delta_values = flat[:, 0]
            grey_values = np.exp(flat[:, 1])
            sigma_values = np.exp(flat[:, 2])
            delta_q = np.quantile(delta_values, (0.025, 0.16, 0.5, 0.84, 0.975))
            shift_q = np.quantile(delta_values - model["reset_day"], (0.025, 0.16, 0.5, 0.84, 0.975))
            grey_q = np.quantile(grey_values, (0.025, 0.16, 0.5, 0.84, 0.975))
            sigma_q = np.quantile(sigma_values, (0.16, 0.5, 0.84))
            writer.writerow(
                dict(
                    zip(
                        fields,
                        [
                            segment["detector"],
                            segment["led"],
                            *delta_q,
                            *shift_q,
                            *grey_q,
                            *sigma_q,
                        ],
                    )
                )
            )


def save_plots(output_dir, model, results, detector=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_detectors = model["n_detectors"]
    figure, axes = plt.subplots(n_detectors, 6, figsize=(20, max(4, 3.0 * n_detectors)), squeeze=False, sharey=False)
    for segment, result in zip(model["segments"], results):
        flat = result["chain"].reshape(-1, 3)
        best = flat[np.argmax(result["log_probability"].reshape(-1))]
        delta_t = best[0]
        grey = np.exp(best[1])
        axis = axes[segment["detector_index"], segment["led"] - 1]
        shift = delta_t - model["reset_day"]
        for tile_index in range(segment["pre_values"].shape[0]):
            axis.plot(segment["pre_days"], segment["pre_values"][tile_index], color="tab:blue", alpha=0.10, linewidth=0.4)
            axis.plot(segment["post_days"] + shift, segment["post_values"][tile_index] / grey, color="tab:green", alpha=0.10, linewidth=0.4)
        axis.plot(segment["pre_days"], np.median(segment["pre_values"], axis=0), color="tab:blue", marker="o", linewidth=1.5)
        axis.plot(segment["post_days"] + shift, np.median(segment["post_values"], axis=0) / grey, color="tab:green", marker="o", linewidth=1.5)
        axis.set_title(f"{segment['detector']} LED {segment['led']}")
        axis.set_xlabel("Time from t=0 (days)")
        axis.grid(alpha=0.2)
    for axis in axes[:, 0]:
        axis.set_ylabel("Corrected current counts")
    figure.suptitle("Independent detector/LED count-space best-fit overlays")
    figure.tight_layout()
    figure.savefig(output_dir / "mcmc_shift_overlay_best_fit_counts.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for segment, result in zip(model["segments"], results):
        flat = result["chain"].reshape(-1, 3)
        label = f"{segment['detector']} LED {segment['led']}"
        axes[0].hist(flat[:, 0], bins=60, histtype="step", linewidth=1.0, label=label)
        axes[1].hist(100.0 * (np.exp(flat[:, 1]) - 1.0), bins=60, histtype="step", linewidth=1.0, label=label)
    axes[0].set_ylabel("Posterior samples")
    axes[1].set_ylabel("Posterior samples")
    axes[1].set_xlabel("Multiplicative scale change (%)")
    axes[0].set_title("delta_t from t=0 (days)")
    axes[0].legend(frameon=False, fontsize=7, ncol=3)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "mcmc_posterior_detector_led.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 6))
    for segment, result in zip(model["segments"], results):
        axis.plot(
            np.exp(result["chain"][:, :, -1]),
            alpha=0.08,
            linewidth=0.25,
            label=f"{segment['detector']} LED {segment['led']}",
        )
    axis.set_xlabel("Production step")
    axis.set_ylabel("Independent sigma (counts)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "mcmc_trace_sigma.png", dpi=160)
    plt.close(figure)


def _initial_segment_center(segment, initial_delta_t, initial_scale):
    lower = segment["reset_day"] + segment["shift_prior_lower"]
    upper = segment["reset_day"] + segment["shift_prior_upper"]
    if initial_delta_t is not None:
        candidates = np.asarray([np.clip(initial_delta_t, lower + 1.0e-8, upper - 1.0e-8)])
    else:
        candidates = np.linspace(lower + 1.0e-8, upper - 1.0e-8, 401)
    prediction = _segment_prediction(segment, candidates)
    weights = segment["post_weights"][None, :, :]
    observed = segment["post_values"][None, :, :]
    if initial_scale is None:
        numerator = np.sum(weights * observed * prediction, axis=(1, 2))
        denominator = np.sum(weights * prediction**2, axis=(1, 2))
        scales = numerator / np.maximum(denominator, 1.0e-30)
    else:
        scales = np.full(len(candidates), float(initial_scale))
    scales = np.clip(scales, np.exp(-1.999), np.exp(1.999))
    residual = observed - scales[:, None, None] * prediction
    weighted_sse = np.sum(weights * residual**2, axis=(1, 2))
    best = int(np.argmin(weighted_sse))
    sigma = max(
        float(np.sqrt(weighted_sse[best] / np.sum(segment["post_weights"]))),
        np.exp(-4.0),
    )
    return np.asarray([candidates[best], np.log(scales[best]), np.log(sigma)])


def _fit_independent_segment(segment, seed, warmup_steps, production_steps, nwalkers, initial_delta_t, initial_scale):
    if nwalkers < 6:
        raise ValueError("nwalkers must be at least 6 for an independent three-parameter fit")
    rng = np.random.default_rng(seed)
    center = _initial_segment_center(segment, initial_delta_t, initial_scale)
    lower = segment["reset_day"] + segment["shift_prior_lower"]
    upper = segment["reset_day"] + segment["shift_prior_upper"]
    widths = np.asarray([max((upper - lower) * 1.0e-3, 1.0e-5), 1.0e-4, 1.0e-3])
    positions = center + rng.normal(size=(nwalkers, 3)) * widths
    positions[:, 0] = np.clip(positions[:, 0], lower + 1.0e-8, upper - 1.0e-8)
    positions[:, 1] = np.clip(positions[:, 1], -1.999, 1.999)
    positions[:, 2] = np.clip(positions[:, 2], -4.999, 7.999)
    np.random.seed(seed)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        3,
        lambda theta: segment_log_probability(segment, theta),
        vectorize=True,
    )
    sampler.run_mcmc(positions, warmup_steps, progress=False, skip_initial_state_check=True)
    sampler.reset()
    sampler.run_mcmc(None, production_steps, progress=False, skip_initial_state_check=True)
    chain = sampler.get_chain()
    try:
        autocorrelation_time = sampler.get_autocorr_time(tol=0)
    except emcee.autocorr.AutocorrError:
        autocorrelation_time = np.full(3, np.nan)
    return {
        "chain": chain,
        "log_probability": sampler.get_log_prob(),
        "acceptance_fraction": sampler.acceptance_fraction,
        "autocorrelation_time": autocorrelation_time,
        "initial_center": center,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True, help="Raw statistics CSV or directory")
    parser.add_argument("--corrections", type=Path, required=True, help="Count-space correction CSV or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detectors", nargs="+", default=None)
    parser.add_argument("--reset-day", type=float, default=RESET_DAY)
    parser.add_argument("--additional-acm-step-repeat", type=int, default=39)
    parser.add_argument("--pre-reset-acm-step-repeat", type=int, default=35)
    parser.add_argument("--initial-delta-t-from-t0", type=float, default=None)
    parser.add_argument(
        "--initial-scale",
        type=float,
        default=None,
        help="Optional common starting scale; otherwise estimate one independently per detector/LED",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--production-steps", type=int, default=PRODUCTION_STEPS)
    parser.add_argument("--nwalkers", type=int, default=NWALKERS)
    args = parser.parse_args()
    if args.initial_scale is not None and (
        args.initial_scale <= 0.0 or not np.isfinite(args.initial_scale)
    ):
        raise ValueError("initial scale must be positive and finite")
    model = load_model_data(
        args.statistics,
        args.corrections,
        args.reset_day,
        args.additional_acm_step_repeat,
        args.pre_reset_acm_step_repeat,
        args.detectors,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, segment in enumerate(model["segments"]):
        print(f"Fitting {segment['detector']} LED {segment['led']} independently")
        results.append(
            _fit_independent_segment(
                segment,
                args.seed + index,
                args.warmup_steps,
                args.production_steps,
                args.nwalkers,
                args.initial_delta_t_from_t0,
                args.initial_scale,
            )
        )
    chains = np.stack([result["chain"] for result in results])
    log_probabilities = np.stack([result["log_probability"] for result in results])
    summary = _independent_summary(
        model,
        results,
        args.initial_delta_t_from_t0,
        args.initial_scale,
    )
    summary.update(
        {
            "seed": args.seed,
            "nwalkers": args.nwalkers,
            "warmup_steps": args.warmup_steps,
            "production_steps_per_walker": args.production_steps,
            "chain_shape": list(chains.shape),
            "n_independent_fits": len(results),
            "acceptance_fraction_mean": float(
                np.mean([np.mean(result["acceptance_fraction"]) for result in results])
            ),
        }
    )
    np.savez(
        output_dir / "posterior_samples.npz",
        chains=chains,
        log_probability=log_probabilities,
        detectors=np.asarray(model["detectors"]),
        detector_led=np.asarray([f"{segment['detector']}_LED{segment['led']}" for segment in model["segments"]]),
        reset_day=model["reset_day"],
    )
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    _write_parameter_csv(output_dir / "detector_led_shift_scale_posterior.csv", model, results)
    (output_dir / "summary.md").write_text(
        "# Independent detector/LED count-space shift/scale MCMC\n\n"
        "Each detector/LED is fitted with its own time shift, multiplicative scale, and noise. "
        "The count-space corrections are applied before interpolation.\n"
    )
    save_plots(output_dir, model, results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
