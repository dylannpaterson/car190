"""Fit a complex-reflectance correction to one current production MAP tile."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from fit_wfi_tmm_production import dielectric_ar_reference
from plot_wfi_tmm_numpyro import (
    _load_prediction_inputs,
    apply_early_time_data_correction,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_FIT_DIR = (
    ROOT
    / "wfi_production_20261005_shared_hgcdte_prior10_smooth001_WFI01_map10000_emcee1000"
)
DEFAULT_STATISTICS = ROOT / "rrn_reduction_20261005/wfi01_tile_log_statistics_512_count_corrected.csv"
DEFAULT_OUTPUT_DIR = ROOT / "wfi01_tile0_complex_q_production_fit"
PARAMETER_NAMES = (
    "log_q_magnitude",
    "q_magnitude_slope",
    "q_phase_offset_rad",
    "q_optical_path_nm",
    "q_absorption_per_nm",
)


def sample_scalar(samples, key, default):
    if key not in samples:
        return default
    values = np.asarray(samples[key])
    if values.size != 1:
        raise ValueError(f"Expected scalar metadata for {key}")
    return values.reshape(-1)[0].item()


def load_production_tile(fit_dir, statistics_path, tile_id):
    with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        detector_names = np.asarray(samples["sca_names"]).astype(str)
        if detector_names.size != 1:
            raise ValueError("The first test must use one detector output")
        detector = detector_names[0]
        posterior_tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)
        tile_indices = np.flatnonzero(posterior_tile_ids == tile_id)
        if tile_indices.size != 1:
            raise ValueError(f"Tile {tile_id} is not uniquely present in the posterior")
        posterior_tile_index = int(tile_indices[0])
        ar_posterior = np.asarray(samples["ar_parameters"], dtype=float)
        if ar_posterior.ndim == 4:
            ar_posterior = ar_posterior.reshape(
                ar_posterior.shape[0], -1, ar_posterior.shape[-1]
            )
        tile_ar_parameters = np.median(
            ar_posterior[:, posterior_tile_index, :], axis=0
        ).reshape(-1, 3)
        tile_rates = np.asarray(
            samples["tile_instantaneous_rate_nm_day"], dtype=float
        )
        if tile_rates.ndim != 3:
            raise ValueError("Unexpected tile instantaneous-rate posterior shape")
        tile_rate = np.median(tile_rates[:, posterior_tile_index, :], axis=0)
        reset_day = float(np.asarray(samples["ice_reset_day"], dtype=float).item())
        sed_mode = str(sample_scalar(samples, "sed_mode", "core"))
        fit_incidence_angle = bool(sample_scalar(samples, "fit_incidence_angle", 0))

    loaded = _load_prediction_inputs(
        fit_dir,
        statistics_path,
        Path("/dev/null"),
        sed_mode,
        1,
        None,
        1,
    )
    (
        data,
        context,
        ice_thickness,
        ar_parameters,
        acm_step,
        hgcdte_parameters,
        ice_modulation_parameters,
        fit_hgcdte,
        conductive_parameters,
        fit_conductive_layer,
        qe_rbf_parameters,
        fit_qe_rbf,
        jax,
        jnp,
        build_prediction_function,
        early_time_scale,
        early_time_cutoff_day,
        late_acm_step,
        additional_acm_step,
        detector_thickness,
    ) = loaded
    if data["n_scas"] != 1 or data["sca_names"].astype(str).tolist() != [detector]:
        raise ValueError(f"Expected one {detector} SCA")
    if data["tile_ids"].size != 1 or int(data["tile_ids"][0]) != tile_id:
        raise ValueError(
            f"Expected selected tile {tile_id}; loaded tile IDs {data['tile_ids']}"
        )

    med = lambda value: None if value is None else np.median(value, axis=0)
    ice_median = med(ice_thickness)
    ar_median = med(ar_parameters)
    acm_median = med(acm_step)
    hgcdte_median = med(hgcdte_parameters)
    conductive_median = med(conductive_parameters)
    qe_median = med(qe_rbf_parameters)
    early_median = med(early_time_scale)
    late_median = med(late_acm_step)
    additional_median = med(additional_acm_step)
    detector_thickness_median = med(detector_thickness)
    incidence_angle = (
        np.asarray(context["incidence_angle_deg"], dtype=float)
        if fit_incidence_angle
        else None
    )

    predictor = jax.jit(
        build_prediction_function(
            data,
            context,
            row_indices=np.arange(data["counts"].shape[0]),
            fit_hgcdte=fit_hgcdte,
            fit_hgcdte_modulation=bool(context["fit_hgcdte_modulation"]),
            fit_conductive_layer=fit_conductive_layer,
            finite_hgcdte_detector=context["finite_hgcdte_detector"],
            detector_backside_index=context["detector_backside_index"],
            fit_qe_rbf=fit_qe_rbf,
            qe_rbf_node_wavelengths_nm=context["qe_rbf_node_wavelengths_nm"],
            qe_rbf_width_nm=context["qe_rbf_width_nm"],
            use_asw_optical_constants=context["use_asw_optical_constants"],
            use_hgcdte_optical_constants=context["use_hgcdte_optical_constants"],
            incidence_cone_half_angle_deg=context["incidence_cone_half_angle_deg"],
            incidence_pupil_half_angle_deg=context["incidence_pupil_half_angle_deg"],
            incidence_chief_ray_tilts_deg=context["incidence_chief_ray_tilts_deg"],
            fit_incidence_angle=fit_incidence_angle,
            share_ar_across_rows=bool(context["shared_ar_parameters_across_tiles"]),
            fit_late_acm_step=late_median is not None,
            fit_additional_acm_step=additional_median is not None,
            absolute_substrate_power=bool(context["absolute_substrate_power"]),
        )
    )
    baseline_prediction = np.asarray(
        predictor(
            jnp.asarray(ice_median),
            jnp.asarray(ar_median),
            jnp.asarray(acm_median),
            None if hgcdte_median is None else jnp.asarray(hgcdte_median),
            None if conductive_median is None else jnp.asarray(conductive_median),
            None if qe_median is None else jnp.asarray(qe_median),
            None if early_median is None else jnp.asarray(early_median),
            None if late_median is None else jnp.asarray(late_median),
            None
            if detector_thickness_median is None
            else jnp.asarray(detector_thickness_median),
            None if incidence_angle is None else jnp.asarray(incidence_angle),
            None if additional_median is None else jnp.asarray(additional_median),
        )
    )
    corrected_data, baseline_prediction = apply_early_time_data_correction(
        data,
        baseline_prediction,
        early_time_scale,
        early_time_cutoff_day,
    )
    counts = np.asarray(corrected_data["counts"], dtype=float).reshape(-1)
    observed = (
        np.asarray(corrected_data["sums"], dtype=float) / counts[None, :]
    ).reshape(-1)
    baseline_prediction = np.asarray(baseline_prediction, dtype=float).reshape(-1)
    return {
        "detector": detector,
        "tile_id": tile_id,
        "data": corrected_data,
        "context": context,
        "ice_thickness": np.asarray(ice_median, dtype=float).reshape(-1),
        "instantaneous_rate": tile_rate,
        "ar_parameters": tile_ar_parameters,
        "reset_day": reset_day,
        "baseline_prediction": baseline_prediction,
        "counts": counts,
        "observed": observed,
    }


def q_spectrum(parameters, wavelengths):
    (
        log_magnitude,
        magnitude_slope,
        phase_offset,
        optical_path_nm,
        absorption_per_nm,
    ) = parameters
    x = (wavelengths - 1400.0) / 1000.0
    magnitude = np.exp(log_magnitude + magnitude_slope * x)
    phase = phase_offset + 2.0 * np.pi * optical_path_nm * (
        1.0 / wavelengths - 1.0 / 1400.0
    )
    return magnitude * np.exp(1j * phase), absorption_per_nm


def broadband_q_response(state, parameters, thicknesses):
    wavelengths = np.asarray(state["context"]["wavelengths"], dtype=float)
    q, absorption_per_nm = q_spectrum(parameters, wavelengths)
    thicknesses = np.asarray(thicknesses, dtype=float).reshape(-1)
    phase = 4.0 * np.pi * 1.25 * thicknesses[:, None] / wavelengths[None, :]
    numerator = np.abs(1.0 - q[None, :]) ** 2
    denominator = np.abs(1.0 - q[None, :] * np.exp(1j * phase)) ** 2
    factors = np.exp(-absorption_per_nm * thicknesses[:, None]) * numerator / denominator
    weights = np.asarray(state["context"]["model_weights"], dtype=float)
    clean = np.asarray(state["context"]["clean_output"], dtype=float)
    if weights.ndim == 3:
        weights = weights[0]
    if clean.ndim == 2:
        clean = clean[0]
    response = factors @ weights.T
    response /= clean[None, :]
    if not np.all(np.isfinite(response)) or np.any(response <= 0.0):
        raise ValueError("Complex-q response is not finite and positive")
    return response


def q_log_correction(state, parameters):
    data = state["data"]
    thickness = state["ice_thickness"]
    led = np.asarray(data["led"], dtype=int) - 1
    reference_indices = np.asarray(data["first_by_led"], dtype=int)
    reference_thickness = thickness[reference_indices]
    response = broadband_q_response(
        state,
        parameters,
        np.concatenate((thickness, reference_thickness)),
    )
    group_indices = np.arange(len(thickness))
    group_response = response[group_indices, led]
    reference_response = response[len(thickness) + led, led]
    return np.log(group_response / reference_response)


def fit_statistics(state, prediction):
    residual = np.asarray(state["observed"] - prediction, dtype=float)
    counts = state["counts"]
    return {
        "unweighted_rms_log": float(np.sqrt(np.mean(residual**2))),
        "count_weighted_rms_log": float(
            np.sqrt(np.average(residual**2, weights=counts))
        ),
        "mean_residual_log": float(np.mean(residual)),
        "count_weighted_sse": float(np.sum(counts * residual**2)),
    }


def fit_complex_q(state, starts, max_nfev):
    counts = state["counts"]
    baseline = state["baseline_prediction"]
    observed = state["observed"]
    scale = np.sqrt(counts / np.median(counts))
    lower = np.asarray([np.log(1.0e-5), -1.0, -np.pi, -2000.0, 0.0])
    upper = np.asarray([np.log(0.2), 1.0, np.pi, 2000.0, 0.02])
    fits = []

    def residual(parameters):
        correction = q_log_correction(state, parameters)
        return scale * (baseline + correction - observed)

    for start in starts:
        result = least_squares(
            residual,
            np.asarray(start, dtype=float),
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        fits.append(result)
    return min(fits, key=lambda result: float(np.sum(result.fun**2)))


def write_predictions(path, state, q_prediction):
    data = state["data"]
    fields = (
        "group",
        "led",
        "repeat",
        "days",
        "observed_log_change",
        "production_log_change",
        "complex_q_log_change",
        "production_residual_log",
        "complex_q_residual_log",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(state["observed"])):
            writer.writerow(
                {
                    "group": int(data["groups"][index]),
                    "led": int(data["led"][index]),
                    "repeat": int(data["repeat"][index]),
                    "days": float(data["days"][index]),
                    "observed_log_change": state["observed"][index],
                    "production_log_change": state["baseline_prediction"][index],
                    "complex_q_log_change": q_prediction[index],
                    "production_residual_log": state["observed"][index]
                    - state["baseline_prediction"][index],
                    "complex_q_residual_log": state["observed"][index]
                    - q_prediction[index],
                }
            )


def write_q_spectrum(path, state, parameters):
    wavelengths = np.asarray(state["context"]["wavelengths"], dtype=float)
    q, absorption_per_nm = q_spectrum(parameters, wavelengths)
    with path.open("w", newline="") as handle:
        fields = (
            "wavelength_nm",
            "q_real",
            "q_imag",
            "q_magnitude",
            "q_phase_rad",
            "q_absorption_per_nm",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for wavelength, value in zip(wavelengths, q):
            writer.writerow(
                {
                    "wavelength_nm": wavelength,
                    "q_real": value.real,
                    "q_imag": value.imag,
                    "q_magnitude": abs(value),
                    "q_phase_rad": np.angle(value),
                    "q_absorption_per_nm": absorption_per_nm,
                }
            )


def write_state_csvs(output_dir, state):
    ar_path = output_dir / "ar_parameters.csv"
    with ar_path.open("w", newline="") as handle:
        fields = (
            "layer",
            "thickness_scale",
            "thickness_nm",
            "refractive_index_n",
            "absorption_index_k",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer, (scale, index_n, index_k) in enumerate(
            state["ar_parameters"], start=1
        ):
            writer.writerow(
                {
                    "layer": layer,
                    "thickness_scale": scale,
                    "thickness_nm": 100.0 * scale,
                    "refractive_index_n": index_n,
                    "absorption_index_k": index_k,
                }
            )

    trajectory_path = output_dir / "ice_trajectory.csv"
    with trajectory_path.open("w", newline="") as handle:
        fields = ("days", "ice_thickness_nm", "instantaneous_rate_nm_day")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for day, thickness, rate in zip(
            state["data"]["days"],
            state["ice_thickness"],
            state["instantaneous_rate"],
        ):
            writer.writerow(
                {
                    "days": day,
                    "ice_thickness_nm": thickness,
                    "instantaneous_rate_nm_day": rate,
                }
            )


def plot_ice_trajectory(path, state):
    days = np.asarray(state["data"]["days"], dtype=float)
    thickness = np.asarray(state["ice_thickness"], dtype=float)
    rate = np.asarray(state["instantaneous_rate"], dtype=float)
    reset_day = state["reset_day"]
    figure, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, constrained_layout=True
    )
    for axis, values, ylabel in (
        (axes[0], thickness, "Ice thickness (nm)"),
        (axes[1], rate, "Instantaneous rate (nm/day)"),
    ):
        for mask, color, label in (
            (days < reset_day, "tab:blue", "pre-reset"),
            (days >= reset_day, "tab:orange", "post-reset"),
        ):
            axis.plot(
                days[mask],
                values[mask],
                "o-",
                ms=2.5,
                lw=1.0,
                color=color,
                label=label,
            )
        axis.axvline(
            reset_day,
            color="black",
            linestyle=":",
            label="reset" if axis is axes[0] else None,
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].set_title(
        f"{state['detector']} tile {state['tile_id']} fitted production ice trajectory"
    )
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Days since first exposure")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_ar_parameters(path, state):
    parameters = np.asarray(state["ar_parameters"], dtype=float)
    reference = dielectric_ar_reference(parameters.shape[0]).reshape(-1, 3)
    layers = np.arange(1, len(parameters) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    specs = (
        (0, "Thickness (nm)", 100.0, 1.0, 1000.0, "thickness"),
        (1, "Refractive index n", 1.0, 1.0, 4.0, "n"),
        (2, "Absorption index k", 1.0, 0.0, 0.5, "k"),
    )
    for column, (parameter, ylabel, scale, lower, upper, label) in enumerate(specs):
        values = parameters[:, parameter] * scale
        prior = reference[:, parameter] * scale
        axes[column].plot(layers, values, "o-", label="tile-0 MAP")
        axes[column].plot(layers, prior, "s--", label="stack prior center")
        axes[column].axhline(lower, color="black", linewidth=0.8, linestyle=":")
        axes[column].axhline(upper, color="black", linewidth=0.8, linestyle=":")
        axes[column].set_xticks(layers)
        axes[column].set_xlabel("AR layer")
        axes[column].set_ylabel(ylabel)
        axes[column].set_ylim(
            lower - 0.05 * (upper - lower), upper + 0.05 * (upper - lower)
        )
        axes[column].grid(alpha=0.2)
        axes[column].set_title(label)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"{state['detector']} tile {state['tile_id']} fitted AR parameters; production k is fixed to zero",
        fontsize=13,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_summary(path, state, result, baseline_stats, complex_stats):
    parameters = result.x
    q_magnitude = float(np.exp(parameters[0]))
    lines = [
        "# WFI single-tile production complex-q fit",
        "",
        "This is a first response-model sensitivity fit to the current production data, not the legacy SCA1 effective-q table.",
        "The archived production MAP optical and ice state is held fixed; only the bounded complex round-trip correction is fitted.",
        "",
        f"Detector: `{state['detector']}`",
        f"Tile: `{state['tile_id']}`",
        f"Observations: `{len(state['observed'])}`",
        "",
        "| Quantity | Production MAP | Production + fitted complex q |",
        "|---|---:|---:|",
        f"| Unweighted RMS log | {baseline_stats['unweighted_rms_log']:.10g} | {complex_stats['unweighted_rms_log']:.10g} |",
        f"| Count-weighted RMS log | {baseline_stats['count_weighted_rms_log']:.10g} | {complex_stats['count_weighted_rms_log']:.10g} |",
        f"| Count-weighted SSE | {baseline_stats['count_weighted_sse']:.10g} | {complex_stats['count_weighted_sse']:.10g} |",
        "",
        "## Fitted complex-q parameters",
        "",
        f"- q magnitude: `{q_magnitude:.10g}`",
        f"- q magnitude slope: `{parameters[1]:.10g}` per normalized wavelength",
        f"- q phase offset: `{parameters[2]:.10g}` rad",
        f"- q optical path: `{parameters[3]:.10g}` nm",
        f"- q absorption: `{parameters[4]:.10g}` per nm",
        f"- optimizer status: `{result.status}` ({result.message})",
        f"- optimizer evaluations: `{result.nfev}`",
        "",
        "The correction is physically bounded by `0 < |q| < 0.9` and non-negative attenuation, but q remains an effective boundary parameter rather than a uniquely identified coating layer.",
        "",
        "## Archived production AR state",
        "",
        "The AR parameters are an effective three-layer MAP state. Production fixes `k=0`; layer 3 is close to the upper allowed refractive-index bound, so these values should not be interpreted as measured coating materials.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_fit(path, state, q_prediction, complex_stats):
    data = state["data"]
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    days = np.asarray(data["days"], dtype=float)
    for led in range(1, 7):
        indices = np.flatnonzero(np.asarray(data["led"], dtype=int) == led)
        color = colors(led - 1)
        axes[0].plot(days[indices], 100.0 * np.expm1(state["observed"][indices]), "o", ms=3, color=color, alpha=0.7)
        axes[0].plot(days[indices], 100.0 * np.expm1(state["baseline_prediction"][indices]), "--", color=color, alpha=0.8, label=f"LED {led} production")
        axes[0].plot(days[indices], 100.0 * np.expm1(q_prediction[indices]), color=color, label=f"LED {led} complex q")
    axes[0].set_ylabel("Reference-normalized change (%)")
    axes[0].set_title("WFI01 tile 0: current production versus fitted complex-q correction")
    axes[0].legend(ncol=3, fontsize=8, frameon=False)
    residual = state["observed"] - q_prediction
    for led in range(1, 7):
        indices = np.flatnonzero(np.asarray(data["led"], dtype=int) == led)
        axes[1].plot(days[indices], residual[indices], "o-", ms=2.5, label=f"LED {led}")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Days since first exposure")
    axes[1].set_ylabel("Complex-q residual (log)")
    axes[1].set_title(
        f"Residual; count-weighted RMS = {complex_stats['count_weighted_rms_log']:.6g}"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT_DIR)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--tile-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-nfev", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = load_production_tile(args.fit_dir, args.statistics, args.tile_id)
    baseline_stats = fit_statistics(state, state["baseline_prediction"])
    starts = (
        (np.log(0.01), 0.0, 0.0, 0.0, 0.0),
        (np.log(0.03), 0.3, 1.0, 100.0, 0.0),
        (np.log(0.05), -0.3, -1.0, -100.0, 0.0001),
        (np.log(0.1), 0.5, 2.0, 300.0, 0.0005),
    )
    result = fit_complex_q(state, starts, args.max_nfev)
    correction = q_log_correction(state, result.x)
    q_prediction = state["baseline_prediction"] + correction
    complex_stats = fit_statistics(state, q_prediction)
    q, absorption_per_nm = q_spectrum(
        result.x, np.asarray(state["context"]["wavelengths"], dtype=float)
    )
    summary = {
        "detector": state["detector"],
        "tile_id": state["tile_id"],
        "parameter_names": PARAMETER_NAMES,
        "parameters": result.x.tolist(),
        "q_magnitude": float(np.exp(result.x[0])),
        "q_magnitude_max": float(np.max(np.abs(q))),
        "q_absorption_per_nm": float(absorption_per_nm),
        "baseline": baseline_stats,
        "complex_q": complex_stats,
        "improvement_in_count_weighted_rms_log": baseline_stats[
            "count_weighted_rms_log"
        ]
        - complex_stats["count_weighted_rms_log"],
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
        "optimizer_nfev": int(result.nfev),
    }
    (args.output_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_predictions(args.output_dir / "predictions.csv", state, q_prediction)
    write_q_spectrum(args.output_dir / "complex_q_spectrum.csv", state, result.x)
    write_state_csvs(args.output_dir, state)
    write_summary(args.output_dir / "fit_summary.md", state, result, baseline_stats, complex_stats)
    plot_fit(args.output_dir / "complex_q_fit.png", state, q_prediction, complex_stats)
    plot_ice_trajectory(args.output_dir / "ice_trajectory.png", state)
    plot_ar_parameters(args.output_dir / "ar_parameters.png", state)
    print(
        f"Production RMS={baseline_stats['count_weighted_rms_log']:.8g}; "
        f"complex-q RMS={complex_stats['count_weighted_rms_log']:.8g}; "
        f"improvement={summary['improvement_in_count_weighted_rms_log']:.8g}"
    )
    print(f"Fitted |q| at 1400 nm={summary['q_magnitude']:.8g}; OPD={result.x[3]:.8g} nm")
    print(f"Wrote {args.output_dir / 'fit_summary.md'}")


if __name__ == "__main__":
    main()
