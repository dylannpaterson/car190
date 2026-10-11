"""Screen scalar and wavelength-dependent real ice indices on a production MAP tile.

This is an opt-in diagnostic. It keeps the production MAP AR, thickness,
HgCdTe, angle, and grey-offset state fixed, and only refits the real ice index.
The result is therefore a screening test for spectral leverage, not a
replacement production fit.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import fit_wfi_tmm_production as production


INDEX_MEAN = 1.25
INDEX_PRIOR_SIGMA = 0.03
INDEX_LOWER = 1.15
INDEX_UPPER = 1.40
INDEX_SLOPE_PRIOR_SIGMA = 0.03
INDEX_SLOPE_LOWER = -0.20
INDEX_SLOPE_UPPER = 0.20
INDEX_PIVOT_NM = 1050.0
INDEX_SCALE_NM = 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=200)
    return parser.parse_args()


def load_map_state(fit_dir):
    fit_dir = Path(fit_dir)
    samples_path = (
        fit_dir
        if fit_dir.is_file()
        else fit_dir / "posterior_samples.npz"
    )
    with np.load(samples_path, allow_pickle=False) as samples:
        def first_available(*names):
            for name in names:
                if name in samples:
                    return samples[name]
            raise KeyError(f"None of the archived sites are present: {names}")

        def scalar_site(name):
            return float(np.asarray(samples[name], dtype=float).reshape(-1)[0])

        return {
            "ar_parameters": np.asarray(samples["ar_parameters"][0], dtype=float).reshape(1, -1),
            "ice_thickness": np.asarray(
                first_available("ice_thickness", "ice")[0], dtype=float
            ).reshape(1, -1),
            "hgcdte_parameters": np.column_stack(
                (
                    np.asarray(samples["absolute_hgcdte_n_log_scale"][0], dtype=float).reshape(-1),
                    np.asarray(samples["absolute_hgcdte_k_log_scale"][0], dtype=float).reshape(-1),
                )
            ),
            "incidence_angle_deg": scalar_site("incidence_angle_deg"),
            "post_reset_tile_log_offset": scalar_site("post_reset_tile_log_offset"),
        }


def build_index_predictor(data, context, state):
    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    wavelengths = jnp.asarray(context["wavelengths"])
    phase_factor = 2.0 * jnp.pi / wavelengths
    model_weights = jnp.asarray(context["model_weights"])[0]
    led_index = jnp.asarray(np.asarray(data["led"], dtype=int) - 1)
    selected_weights = model_weights[led_index]
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    base_n = jnp.asarray(context["hgcdte_n"])
    base_k = jnp.asarray(context["hgcdte_k"])
    ar_parameters = jnp.asarray(state["ar_parameters"])
    ice_thickness = jnp.asarray(state["ice_thickness"])
    hgcdte_parameters = jnp.asarray(state["hgcdte_parameters"])
    angle_sine = jnp.broadcast_to(
        production._incidence_sine(state["incidence_angle_deg"], jnp), (1,)
    )
    wavelength_coordinate = (wavelengths - INDEX_PIVOT_NM) / INDEX_SCALE_NM

    baseline_offsets = jnp.asarray(
        [
            np.average(
                data["sums"][:, group] / data["counts"][:, group],
                weights=data["counts"][:, group],
            )
            for group in np.asarray(data["first_by_led"], dtype=int)
        ]
    )

    def layer_elements(index, thickness, polarization):
        cosine = jnp.sqrt(1.0 - (angle_sine[:, None] / index) ** 2)
        admittance = index * cosine if polarization == "s" else index / cosine
        phase = (
            phase_factor[None, None, :]
            * index[:, :, None]
            * thickness[:, :, None]
            * cosine[:, :, None]
        )
        cosine_phase = jnp.cos(phase)
        sine_phase = -1j * jnp.sin(phase)
        return (
            cosine_phase,
            sine_phase / admittance[:, :, None],
            sine_phase * admittance[:, :, None],
            cosine_phase,
        )

    def prediction(parameters):
        n0 = parameters[0]
        slope = parameters[1]
        ice_index = n0 + slope * wavelength_coordinate + 0.0j
        ar = ar_parameters.reshape((1, production.AR_LAYERS, 3))
        ar_index = ar[:, :, 1] + 1j * ar[:, :, 2]
        ar_thickness = 100.0 * ar[:, :, 0]
        substrate = (
            base_n[None, :] * jnp.exp(hgcdte_parameters[:, 0, None])
            + 1j * base_k[None, :] * jnp.exp(hgcdte_parameters[:, 1, None])
        )

        def polarization_response(polarization):
            incident_cosine = jnp.sqrt(1.0 - angle_sine**2)
            incident_admittance = (
                incident_cosine
                if polarization == "s"
                else 1.0 / incident_cosine
            )
            substrate_cosine = jnp.sqrt(
                1.0 - (angle_sine[:, None] / substrate) ** 2
            )
            substrate_admittance = (
                substrate * substrate_cosine
                if polarization == "s"
                else substrate / substrate_cosine
            )
            boundary_first = jnp.ones(
                (ar.shape[0], wavelengths.size), dtype=jnp.complex128
            )
            boundary_second = substrate_admittance
            for layer_index in range(production.AR_LAYERS - 1, -1, -1):
                elements = layer_elements(
                    ar_index[:, layer_index : layer_index + 1],
                    ar_thickness[:, layer_index : layer_index + 1],
                    polarization,
                )
                old_first, old_second = boundary_first, boundary_second
                boundary_first = (
                    elements[0][:, 0, :] * old_first
                    + elements[1][:, 0, :] * old_second
                )
                boundary_second = (
                    elements[2][:, 0, :] * old_first
                    + elements[3][:, 0, :] * old_second
                )
            clean_denominator = (
                incident_admittance[:, None] * boundary_first + boundary_second
            )
            clean = (
                jnp.real(substrate_admittance / incident_admittance[:, None])
                * jnp.abs(2.0 * incident_admittance[:, None] / clean_denominator) ** 2
            )

            ice_cosine = jnp.sqrt(
                1.0 - (angle_sine[:, None, None] / ice_index[None, None, :]) ** 2
            )
            ice_admittance = (
                ice_index[None, None, :] * ice_cosine
                if polarization == "s"
                else ice_index[None, None, :] / ice_cosine
            )
            ice_phase = (
                phase_factor[None, None, :]
                * ice_thickness[:, :, None]
                * ice_index[None, None, :]
                * ice_cosine
            )
            ice_cosine_phase = jnp.cos(ice_phase)
            ice_sine_phase = -1j * jnp.sin(ice_phase)
            ice_m01 = ice_sine_phase / ice_admittance
            ice_m10 = ice_sine_phase * ice_admittance
            coated_first = (
                ice_cosine_phase * boundary_first[:, None, :]
                + ice_m01 * boundary_second[:, None, :]
            )
            coated_second = (
                ice_m10 * boundary_first[:, None, :]
                + ice_cosine_phase * boundary_second[:, None, :]
            )
            coated_denominator = (
                incident_admittance[:, None, None] * coated_first + coated_second
            )
            coated = (
                jnp.real(
                    substrate_admittance[:, None, :]
                    / incident_admittance[:, None, None]
                )
                * jnp.abs(
                    2.0 * incident_admittance[:, None, None] / coated_denominator
                )
                ** 2
            )
            return clean, coated

        clean_s, coated_s = polarization_response("s")
        clean_p, coated_p = polarization_response("p")
        clean = 0.5 * (clean_s + clean_p)
        coated = 0.5 * (coated_s + coated_p)
        integrated_clean = jnp.sum(clean[:, None, :] * selected_weights[None, :, :], axis=-1)
        integrated = jnp.sum(coated * selected_weights[None, :, :], axis=-1)
        current = integrated / integrated_clean
        reference = current[:, first_by_led]
        baseline = reference[:, led_index]
        return jnp.log(current / baseline) + baseline_offsets[led_index][None, :]

    return jax, jnp, prediction


def run_screen(data, predictor, state, wavelengths, mode, max_nfev):
    jax, jnp = predictor[0], predictor[1]
    prediction = predictor[2]
    counts = jnp.asarray(data["counts"])
    observed = jnp.asarray(data["sums"]) / jnp.maximum(counts, 1.0)
    observed = observed + state["post_reset_tile_log_offset"] * jnp.asarray(
        np.asarray(data["days"]) >= float(data["reset_day"])
    )[None, :]
    likelihood_scale = np.sqrt(
        production.default_likelihood_weight(data["counts"])
        * np.asarray(data["counts"], dtype=float)
    ) / production.LOG_NOISE_SIGMA

    def expand(parameters):
        if mode == "scalar":
            return jnp.asarray([parameters[0], 0.0])
        return parameters

    def residual(parameters):
        data_residual = (
            prediction(expand(parameters)) - observed
        ) * jnp.asarray(likelihood_scale)
        prior = [(parameters[0] - INDEX_MEAN) / INDEX_PRIOR_SIGMA]
        if mode == "linear":
            prior.append(parameters[1] / INDEX_SLOPE_PRIOR_SIGMA)
        return jnp.concatenate((data_residual.reshape(-1), jnp.asarray(prior)))

    residual_jit = jax.jit(residual)
    jacobian_jit = jax.jit(jax.jacfwd(residual))

    def scipy_residual(parameters):
        return np.asarray(residual_jit(jnp.asarray(parameters)), dtype=float)

    def scipy_jacobian(parameters):
        return np.asarray(jacobian_jit(jnp.asarray(parameters)), dtype=float)

    if mode == "scalar":
        initial = np.asarray([INDEX_MEAN], dtype=float)
        lower = np.asarray([INDEX_LOWER], dtype=float)
        upper = np.asarray([INDEX_UPPER], dtype=float)
    else:
        initial = np.asarray([INDEX_MEAN, 0.0], dtype=float)
        lower = np.asarray([INDEX_LOWER, INDEX_SLOPE_LOWER], dtype=float)
        upper = np.asarray([INDEX_UPPER, INDEX_SLOPE_UPPER], dtype=float)
    result = least_squares(
        scipy_residual,
        initial,
        jac=scipy_jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=max_nfev,
        verbose=0,
    )
    fitted = np.asarray(result.x, dtype=float)
    if mode == "scalar":
        fitted_full = np.asarray([fitted[0], 0.0])
    else:
        fitted_full = fitted
    fitted_prediction = np.asarray(prediction(jnp.asarray(fitted_full)), dtype=float).reshape(-1)
    observed_array = np.asarray(observed, dtype=float).reshape(-1)
    raw_residual = fitted_prediction - observed_array
    data_residual = raw_residual * likelihood_scale.reshape(-1)
    prior_residual = scipy_residual(fitted)[-len(fitted) :]
    wavelengths = np.asarray(wavelengths, dtype=float)
    wavelength_min = float(np.min(wavelengths))
    wavelength_max = float(np.max(wavelengths))
    return {
        "mode": mode,
        "parameters": fitted,
        "n_at_min_wavelength": float(
            fitted_full[0]
            + fitted_full[1] * (wavelength_min - INDEX_PIVOT_NM) / INDEX_SCALE_NM
        ),
        "n_at_max_wavelength": float(
            fitted_full[0]
            + fitted_full[1] * (wavelength_max - INDEX_PIVOT_NM) / INDEX_SCALE_NM
        ),
        "raw_rmse": float(np.sqrt(np.mean(raw_residual**2))),
        "weighted_rmse": float(np.sqrt(np.mean(data_residual**2))),
        "data_cost": float(0.5 * np.sum(data_residual**2)),
        "prior_cost": float(0.5 * np.sum(prior_residual**2)),
        "joint_cost": float(result.cost),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "prediction": fitted_prediction,
        "observed": observed_array,
        "residual": raw_residual,
    }


def write_outputs(output_dir, data, context, state, results, baseline_prediction):
    output_dir.mkdir(parents=True, exist_ok=True)
    wavelengths = np.asarray(context["wavelengths"], dtype=float)
    rows = []
    for result in results:
        parameters = result["parameters"]
        rows.append(
            {
                "model": result["mode"],
                "n0": float(parameters[0]),
                "slope_per_1000nm": (
                    float(parameters[1]) if len(parameters) > 1 else 0.0
                ),
                "n_at_min_wavelength": result["n_at_min_wavelength"],
                "n_at_max_wavelength": result["n_at_max_wavelength"],
                "raw_rmse": result["raw_rmse"],
                "weighted_rmse": result["weighted_rmse"],
                "data_cost": result["data_cost"],
                "prior_cost": result["prior_cost"],
                "joint_cost": result["joint_cost"],
                "success": result["success"],
                "nfev": result["nfev"],
            }
        )
    with (output_dir / "ice_index_fit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "ice_index_by_wavelength.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["wavelength_nm", "scalar_n", "linear_n"])
        writer.writeheader()
        scalar = results[0]["parameters"][0]
        linear = results[1]["parameters"]
        for wavelength in wavelengths:
            writer.writerow(
                {
                    "wavelength_nm": float(wavelength),
                    "scalar_n": float(scalar),
                    "linear_n": float(
                        linear[0]
                        + linear[1] * (wavelength - INDEX_PIVOT_NM) / INDEX_SCALE_NM
                    ),
                }
            )

    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(days, baseline_prediction, color="#777777", linewidth=1.0, label="Fixed n=1.25")
    axes[0].plot(days, results[0]["prediction"], color="#0072B2", linewidth=1.1, label="Fitted scalar n")
    axes[0].plot(days, results[1]["prediction"], color="#D55E00", linewidth=1.1, label="Fitted linear n(lambda)")
    axes[0].scatter(days, results[1]["observed"], c=leds, cmap="viridis", s=8, label="Observed")
    axes[0].set_ylabel("Log response")
    axes[0].set_title("WFI ice-index screening at fixed production MAP state")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(days, results[0]["residual"], color="#0072B2", linewidth=0.8, label="Scalar n residual")
    axes[1].plot(days, results[1]["residual"], color="#D55E00", linewidth=0.8, label="Linear n(lambda) residual")
    axes[1].set_xlabel("Days since first exposure")
    axes[1].set_ylabel("Model - observed")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    figure.savefig(output_dir / "ice_index_model_comparison.png", dpi=180)
    plt.close(figure)

    summary = [
        "# Ice-index screening",
        "",
        "- This is a fixed-MAP screening experiment; AR, thickness, HgCdTe, angle, and grey offset were not refit.",
        f"- Wavelength grid: `{wavelengths.size}` samples from `{wavelengths.min():.6g}` to `{wavelengths.max():.6g}` nm.",
        f"- Parameterization: `n(lambda) = n0 + slope * (lambda - {INDEX_PIVOT_NM:g}) / {INDEX_SCALE_NM:g}` nm; `k(lambda)=0`.",
    ]
    for result in results:
        summary.append(
            f"- `{result['mode']}`: n0 `{result['parameters'][0]:.9g}`, slope `{result['parameters'][1] if len(result['parameters']) > 1 else 0.0:.9g}`, "
            f"raw RMSE `{result['raw_rmse']:.9g}`, weighted RMSE `{result['weighted_rmse']:.9g}`, "
            f"data cost `{result['data_cost']:.9g}`, joint cost `{result['joint_cost']:.9g}`."
        )
    (output_dir / "ice_index_fit.md").write_text("\n".join(summary) + "\n")
    with (output_dir / "ice_index_fit.json").open("w") as handle:
        json.dump(
            {
                "wavelength_min_nm": float(wavelengths.min()),
                "wavelength_max_nm": float(wavelengths.max()),
                "baseline_raw_rmse": float(np.sqrt(np.mean((baseline_prediction - results[0]["observed"]) ** 2))),
                "results": [
        {
            key: (value.tolist() if key == "parameters" else value)
            for key, value in result.items()
            if key not in ("prediction", "observed", "residual")
        }
                    for result in results
                ],
            },
            handle,
            indent=2,
        )


def main():
    args = parse_args()
    data = production.load_tile_statistics(args.statistics)
    context = production.load_response_context(data["sca_names"])
    state = load_map_state(args.fit_dir)
    jax, jnp, prediction = build_index_predictor(data, context, state)
    baseline_parameters = jnp.asarray([INDEX_MEAN, 0.0])
    baseline_prediction = np.asarray(prediction(baseline_parameters), dtype=float).reshape(-1)
    results = [
        run_screen(
            data,
            (jax, jnp, prediction),
            state,
            context["wavelengths"],
            "scalar",
            args.max_nfev,
        ),
        run_screen(
            data,
            (jax, jnp, prediction),
            state,
            context["wavelengths"],
            "linear",
            args.max_nfev,
        ),
    ]
    write_outputs(args.output_dir, data, context, state, results, baseline_prediction)
    for result in results:
        print(
            f"{result['mode']}: n0={result['parameters'][0]:.9g}, "
            f"slope={result['parameters'][1] if len(result['parameters']) > 1 else 0.0:.9g}, "
            f"raw_rmse={result['raw_rmse']:.9g}, data_cost={result['data_cost']:.9g}"
        )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
