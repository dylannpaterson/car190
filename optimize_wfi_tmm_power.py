"""Bounded multistart optimization for the constant-rate absolute-power TMM model."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from assess_rcs_spectral_identifiability import QE_URL, SED_URL, load_qe, load_sed
from fit_wfi_tmm_numpyro import (
    AR_LOWER,
    AR_UPPER,
    apply_fixed_corrections,
    dielectric_ar_reference,
    load_tile_statistics,
)
from grid_search_wfi_tmm_power import (
    build_source_integrations,
    predict_constant_rate_log_ratios,
    score_prediction,
)


DEFAULT_STATISTICS = Path(
    "wfi_tile_log_statistics_512/wfi01_tile_log_statistics_512.csv"
)
DEFAULT_FIXED_CORRECTIONS = Path("wfi01_fixed_led3_gain_acm_corrections.csv")
DEFAULT_OPTICAL_START = np.asarray(
    [
        1.033590027361476,
        1.3104470400274995,
        2.3791666469496473,
        1.5088992156170626,
        1.4727145285278207,
        1.9561784870002665,
    ],
    dtype=float,
)


def pack_parameters(
    optical_parameters,
    rate_nm_day,
    initial_thickness_nm=0.0,
    hgcdte_n_scale=1.0,
    hgcdte_k_scale=1.0,
):
    optical_parameters = np.asarray(optical_parameters, dtype=float).reshape(-1)
    if optical_parameters.size != 6:
        raise ValueError("Six optical parameters are required")
    if (
        np.any(optical_parameters[0::2] <= 0.0)
        or rate_nm_day <= 0.0
        or initial_thickness_nm < 0.0
        or hgcdte_n_scale <= 0.0
        or hgcdte_k_scale <= 0.0
    ):
        raise ValueError(
            "AR thickness scales, rate, and HgCdTe scales must be positive; "
            "initial ice thickness must be non-negative"
        )
    packed = optical_parameters.copy()
    packed[0::2] = np.log(packed[0::2])
    return np.concatenate(
        (
            packed,
            [
                np.log(rate_nm_day),
                initial_thickness_nm,
                np.log(hgcdte_n_scale),
                np.log(hgcdte_k_scale),
            ],
        )
    )


def unpack_parameters(packed):
    packed = np.asarray(packed, dtype=float).reshape(-1)
    if packed.size != 10:
        raise ValueError("Ten optimization parameters are required")
    optical_parameters = packed[:6].copy()
    optical_parameters[0::2] = np.exp(optical_parameters[0::2])
    return (
        optical_parameters,
        float(np.exp(packed[6])),
        float(packed[7]),
        float(np.exp(packed[8])),
        float(np.exp(packed[9])),
    )


def optimization_bounds(
    rate_bounds,
    initial_thickness_bounds=(0.0, 100.0),
    hgcdte_n_scale_bounds=(0.5, 3.0),
    hgcdte_k_scale_bounds=(0.1, 5.0),
):
    bounds = []
    for _ in range(3):
        bounds.extend(
            (
                (np.log(AR_LOWER[0]), np.log(AR_UPPER[0])),
                (AR_LOWER[1], AR_UPPER[1]),
            )
        )
    bounds.append((np.log(rate_bounds[0]), np.log(rate_bounds[1])))
    bounds.append(tuple(initial_thickness_bounds))
    bounds.append(
        (
            np.log(hgcdte_n_scale_bounds[0]),
            np.log(hgcdte_n_scale_bounds[1]),
        )
    )
    bounds.append(
        (
            np.log(hgcdte_k_scale_bounds[0]),
            np.log(hgcdte_k_scale_bounds[1]),
        )
    )
    return bounds


def objective(
    packed,
    data,
    source_integrations,
    ice_index,
    use_hgcdte_optical_constants,
):
    (
        optical_parameters,
        rate_nm_day,
        initial_thickness_nm,
        hgcdte_n_scale,
        hgcdte_k_scale,
    ) = unpack_parameters(packed)
    prediction = predict_constant_rate_log_ratios(
        data,
        source_integrations,
        optical_parameters,
        rate_nm_day,
        initial_thickness_nm=initial_thickness_nm,
        ice_index=ice_index,
        use_hgcdte_optical_constants=use_hgcdte_optical_constants,
        hgcdte_n_scale=hgcdte_n_scale,
        hgcdte_k_scale=hgcdte_k_scale,
    )
    score = score_prediction(data, prediction)
    return score["sse"] / np.sum(data["counts"])


def run_optimization(
    data,
    source_integrations,
    starts,
    bounds,
    maxiter,
    ice_index,
    use_hgcdte_optical_constants,
):
    results = []
    for index, start in enumerate(starts, start=1):
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            args=(
                data,
                source_integrations,
                ice_index,
                use_hgcdte_optical_constants,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(maxiter),
                "ftol": 1.0e-12,
                "gtol": 1.0e-7,
                "maxls": 50,
            },
        )
        (
            optical_parameters,
            rate_nm_day,
            initial_thickness_nm,
            hgcdte_n_scale,
            hgcdte_k_scale,
        ) = unpack_parameters(result.x)
        prediction = predict_constant_rate_log_ratios(
            data,
            source_integrations,
            optical_parameters,
            rate_nm_day,
            initial_thickness_nm=initial_thickness_nm,
            ice_index=ice_index,
            use_hgcdte_optical_constants=use_hgcdte_optical_constants,
            hgcdte_n_scale=hgcdte_n_scale,
            hgcdte_k_scale=hgcdte_k_scale,
        )
        score = score_prediction(data, prediction)
        results.append(
            {
                "start": index,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "objective": float(result.fun),
                "optical_parameters": optical_parameters,
                "rate_nm_day": rate_nm_day,
                "initial_thickness_nm": initial_thickness_nm,
                "hgcdte_n_scale": hgcdte_n_scale,
                "hgcdte_k_scale": hgcdte_k_scale,
                **score,
            }
        )
        print(
            f"start {index}/{len(starts)}: objective={result.fun:.8g}; "
            f"group RMS={score['group_rms']:.8g}; success={result.success}"
        )
    results.sort(key=lambda row: row["sse"])
    return results


def write_outputs(output_dir, data, result, all_results, settings):
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = result["optical_parameters"]
    results_path = output_dir / "optimization_starts.csv"
    fields = [
        "start",
        "success",
        "status",
        "iterations",
        "evaluations",
        "objective",
        "sse",
        "pixel_rms",
        "group_rms",
        "rate_nm_day",
        "initial_thickness_nm",
        "hgcdte_n_scale",
        "hgcdte_k_scale",
        "layer1_thickness_scale",
        "layer1_n",
        "layer2_thickness_scale",
        "layer2_n",
        "layer3_thickness_scale",
        "layer3_n",
    ]
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_results:
            writer.writerow(
                {
                    key: row[key]
                    for key in fields[:13]
                }
                | {
                    **{
                        fields[index + 13]: row["optical_parameters"][index]
                        for index in range(6)
                    },
                }
            )

    prediction_path = output_dir / "best_prediction.csv"
    observed = np.asarray(data["sums"], dtype=float) / data["counts"]
    with prediction_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "days",
                "led",
                "observed_log_ratio",
                "predicted_log_ratio",
                "residual_log_ratio",
            ]
        )
        for group, (day, led, prediction) in enumerate(
            zip(data["days"], data["led"], result["prediction"])
        ):
            observed_group = float(np.mean(observed[:, group]))
            writer.writerow(
                [
                    group,
                    f"{day:.12g}",
                    int(led),
                    f"{observed_group:.12g}",
                    f"{prediction:.12g}",
                    f"{observed_group - prediction:.12g}",
                ]
            )

    summary_path = output_dir / "optimization_summary.md"
    lines = [
        "# Absolute-power TMM optimization",
        "",
        "The fit uses the gain/ACM-corrected ratio sufficient statistics, integrates absolute TMM power entering a semi-infinite HgCdTe substrate over the LED SED, and uses a constant ice accumulation rate. Corrected per-LED reference offsets are retained as deterministic data offsets, not fitted parameters.",
        "",
        f"- Fixed corrections: `{settings['fixed_corrections']}`",
        f"- SED mode/weighting: `{settings['sed_mode']}` / `{settings['sed_weighting']}`",
        f"- QE weighting: `{settings['qe_weighted']}`",
        f"- Initial-thickness start (nm): `{settings['initial_thickness_nm']}`",
        f"- Initial-thickness bounds (nm): `{settings['initial_thickness_bounds']}`",
        f"- Rate bounds (nm/day): `{settings['rate_bounds']}`",
        f"- Fitted initial thickness (nm): `{result['initial_thickness_nm']:.12g}`",
        f"- HgCdTe n-scale bounds: `{settings['hgcdte_n_scale_bounds']}`",
        f"- HgCdTe k-scale bounds: `{settings['hgcdte_k_scale_bounds']}`",
        f"- Fitted HgCdTe n scale: `{result['hgcdte_n_scale']:.12g}`",
        f"- Fitted HgCdTe k scale: `{result['hgcdte_k_scale']:.12g}`",
        f"- HgCdTe wavelength-dependent constants: `{settings['use_hgcdte_optical_constants']}`",
        f"- Optimization starts: `{len(all_results)}`",
        "",
        "## Best result",
        "",
        f"- Optimizer success: `{result['success']}` ({result['message']})",
        f"- Iterations/evaluations: `{result['iterations']}/{result['evaluations']}`",
        f"- Pixel RMS: `{result['pixel_rms']:.12g}` log",
        f"- Group RMS: `{result['group_rms']:.12g}` log",
        f"- Constant accumulation rate: `{result['rate_nm_day']:.12g}` nm/day",
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
    for led_value, axis in enumerate(axes.flat, start=1):
        groups = np.flatnonzero(np.asarray(data["led"]) == led_value)
        axis.plot(data["days"][groups], observed[0, groups], "ko", ms=3, label="Corrected data")
        axis.plot(data["days"][groups], result["prediction"][groups], color="#D55E00", label="Absolute-power TMM")
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
    return summary_path, results_path, prediction_path, plot_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--output-dir", type=Path, default=Path("wfi01_tmm_power_optimize"))
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--rate-bounds", default="1,50")
    parser.add_argument("--initial-thickness-bounds", default="0,100")
    parser.add_argument("--hgcdte-n-scale-bounds", default="0.5,3")
    parser.add_argument("--hgcdte-k-scale-bounds", default="0.1,5")
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--sed-weighting", choices=("energy", "photon"), default="energy")
    parser.add_argument("--qe-weighted", action="store_true")
    parser.add_argument("--ice-index", type=float, default=1.25)
    parser.add_argument("--max-tiles", type=int, default=1)
    parser.add_argument(
        "--fixed-corrections",
        type=Path,
        default=DEFAULT_FIXED_CORRECTIONS,
        help="Fixed gain/ACM sidecar applied before optimization.",
    )
    parser.add_argument("--no-hgcdte-optical-constants", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.starts < 1 or args.maxiter < 1:
        raise ValueError("Starts and maxiter must be positive")
    rate_bounds = parse_float_list(args.rate_bounds)
    if len(rate_bounds) != 2 or rate_bounds[0] <= 0.0 or rate_bounds[1] <= rate_bounds[0]:
        raise ValueError("Rate bounds must be two increasing positive values")
    initial_thickness_bounds = parse_float_list(args.initial_thickness_bounds)
    hgcdte_n_scale_bounds = parse_float_list(args.hgcdte_n_scale_bounds)
    hgcdte_k_scale_bounds = parse_float_list(args.hgcdte_k_scale_bounds)
    for name, values in (
        ("initial thickness", initial_thickness_bounds),
        ("HgCdTe n scale", hgcdte_n_scale_bounds),
        ("HgCdTe k scale", hgcdte_k_scale_bounds),
    ):
        if len(values) != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} bounds must contain two finite values")
        if values[1] <= values[0]:
            raise ValueError(f"{name} bounds must be increasing")
    if initial_thickness_bounds[0] < 0.0:
        raise ValueError("Initial thickness lower bound must be non-negative")
    if hgcdte_n_scale_bounds[0] <= 0.0 or hgcdte_k_scale_bounds[0] <= 0.0:
        raise ValueError("HgCdTe scale lower bounds must be positive")
    data = load_tile_statistics(args.statistics, max_tiles=args.max_tiles)
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
    reference = dielectric_ar_reference(3)
    prior_start = reference[[0, 1, 3, 4, 6, 7]]
    start_parameters = [
        (
            DEFAULT_OPTICAL_START,
            initial_thickness_bounds[0],
            np.clip(1.0, *hgcdte_n_scale_bounds),
            np.clip(1.0, *hgcdte_k_scale_bounds),
        ),
        (
            prior_start,
            np.clip(35.0, *initial_thickness_bounds),
            np.clip(1.0, *hgcdte_n_scale_bounds),
            np.clip(1.0, *hgcdte_k_scale_bounds),
        ),
    ]
    rng = np.random.default_rng(20260925)
    while len(start_parameters) < args.starts:
        random_start = np.empty(6, dtype=float)
        random_start[0::2] = np.exp(
            rng.uniform(np.log(AR_LOWER[0]), np.log(AR_UPPER[0]), size=3)
        )
        random_start[1::2] = rng.uniform(AR_LOWER[1], AR_UPPER[1], size=3)
        start_parameters.append(
            (
                random_start,
                rng.uniform(*initial_thickness_bounds),
                np.exp(
                    rng.uniform(
                        np.log(hgcdte_n_scale_bounds[0]),
                        np.log(hgcdte_n_scale_bounds[1]),
                    )
                ),
                np.exp(
                    rng.uniform(
                        np.log(hgcdte_k_scale_bounds[0]),
                        np.log(hgcdte_k_scale_bounds[1]),
                    )
                ),
            )
        )
    rate_start = np.clip(14.0, *rate_bounds)
    starts = [
        pack_parameters(parameters, rate_start, initial_thickness, n_scale, k_scale)
        for parameters, initial_thickness, n_scale, k_scale in start_parameters[: args.starts]
    ]
    settings = {
        "fixed_corrections": str(args.fixed_corrections),
        "sed_mode": args.sed_mode,
        "sed_weighting": args.sed_weighting,
        "qe_weighted": args.qe_weighted,
        "initial_thickness_nm": initial_thickness_bounds[0],
        "initial_thickness_bounds": initial_thickness_bounds,
        "rate_bounds": rate_bounds,
        "hgcdte_n_scale_bounds": hgcdte_n_scale_bounds,
        "hgcdte_k_scale_bounds": hgcdte_k_scale_bounds,
        "use_hgcdte_optical_constants": not args.no_hgcdte_optical_constants,
    }
    results = run_optimization(
        data,
        source_integrations,
        starts,
        optimization_bounds(
            rate_bounds,
            initial_thickness_bounds,
            hgcdte_n_scale_bounds,
            hgcdte_k_scale_bounds,
        ),
        args.maxiter,
        args.ice_index,
        not args.no_hgcdte_optical_constants,
    )
    paths = write_outputs(args.output_dir, data, results[0], results, settings)
    for path in paths:
        print(f"Wrote {path}")
    print(
        f"Best: group RMS={results[0]['group_rms']:.8g}; "
        f"rate={results[0]['rate_nm_day']:.8g} nm/day"
    )


def parse_float_list(value):
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Expected finite comma-separated values")
    return values


if __name__ == "__main__":
    main()
