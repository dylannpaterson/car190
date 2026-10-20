"""Fit WFI01 tile 136 with the WFI10 tile-136 rate trajectory fixed.

This diagnostic keeps the target tile's initial and post-reset thicknesses free,
but fixes every non-parametric instantaneous-rate node to the WFI10 MAP value.
The remaining production optical and nuisance parameters are then reoptimized.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production
from wfi_production_layout import DEFAULT_TILE_SIZE, positive_integer


PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY = 12.0
PRODUCTION_TRAJECTORY_RATE_PRIOR_SIGMA_LOG = 1.5
PRODUCTION_ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG = 0.25
PRODUCTION_AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG = 0.10
PRODUCTION_ICE_N_CONSTANT_PRIOR_SIGMA_LOG = 0.01

VARIABLE_SITES = (
    "ar_tile_raw",
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "post_reset_tile_log_offset",
    "absolute_hgcdte_n_log_scale",
    "absolute_hgcdte_k_log_scale",
    "ice_n_log_constant",
    "ar_n_log_modulation",
    "hgcdte_n_log_modulation",
    "hgcdte_k_log_modulation",
)
SOURCE_OPTICAL_SITES = (
    "ar_tile_raw",
    "absolute_hgcdte_n_log_scale",
    "absolute_hgcdte_k_log_scale",
    "ice_n_log_constant",
    "ar_n_log_modulation",
    "hgcdte_n_log_modulation",
    "hgcdte_k_log_modulation",
)


def _scalar(values, name):
    return float(np.asarray(values[name], dtype=float).reshape(-1)[0])


def _summary_objective(path):
    match = re.search(
        r"^- Objective: `([^`]+)`\.$",
        Path(path).read_text(),
        flags=re.MULTILINE,
    )
    return None if match is None else float(match.group(1))


def _load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _check_matching_observations(target_data, source_data):
    for name in ("days", "led", "repeat", "program"):
        if not np.array_equal(np.asarray(target_data[name]), np.asarray(source_data[name])):
            raise ValueError(f"Target and source observations differ in {name}")
    if not np.allclose(target_data["node_days"], source_data["node_days"], rtol=0.0, atol=1.0e-12):
        raise ValueError("Target and source trajectory node days differ")


def _source_rate_nodes(source_archive, target_data):
    rates = np.asarray(source_archive["node_instantaneous_rate_nm_day"], dtype=float)
    rates = rates.reshape(-1)
    if rates.size != len(target_data["node_days"]):
        raise ValueError(
            "WFI10 node-rate trajectory does not match the target node count: "
            f"{rates.size} != {len(target_data['node_days'])}"
        )
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("WFI10 node-rate trajectory must be positive and finite")
    return rates


def _reduced_least_squares(
    model,
    initial_values,
    variable_names,
    jax,
    jnp,
    *,
    ar_mean_prior_sigma,
    absolute_hgcdte_n_scale_prior_sigma_log,
    ar_hgcdte_modulation_prior_sigma_log,
    hgcdte_n_modulation_prior_sigma_log,
    ice_n_constant_prior_sigma_log,
):
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    variable_initial = {
        name: np.asarray(initial_values[name], dtype=float).copy()
        for name in variable_names
    }
    fixed_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in initial_values.items()
        if name not in variable_names
    }
    flat_initial, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in variable_initial.items()}
    )
    parameter_scales = production.map_parameter_scale_values(
        initial_values,
        trajectory_rate_prior_sigma_log=PRODUCTION_TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            absolute_hgcdte_n_scale_prior_sigma_log
        ),
        ice_n_constant_prior_sigma_log=ice_n_constant_prior_sigma_log,
        ar_mean_prior_sigma=ar_mean_prior_sigma,
        ar_hgcdte_modulation_prior_sigma_log=ar_hgcdte_modulation_prior_sigma_log,
        hgcdte_n_modulation_prior_sigma_log=hgcdte_n_modulation_prior_sigma_log,
    )
    flat_scales, _ = ravel_pytree(
        {
            name: jnp.asarray(parameter_scales[name])
            for name in variable_names
        }
    )

    lower = {
        name: np.full(np.asarray(value).shape, -np.inf)
        for name, value in variable_initial.items()
    }
    upper = {
        name: np.full(np.asarray(value).shape, np.inf)
        for name, value in variable_initial.items()
    }
    flat_lower, _ = ravel_pytree(
        {name: jnp.asarray(lower[name]) for name in variable_names}
    )
    flat_upper, _ = ravel_pytree(
        {name: jnp.asarray(upper[name]) for name in variable_names}
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)
    scaled_lower = (np.asarray(flat_lower, dtype=float) - flat_initial) / flat_scales
    scaled_upper = (np.asarray(flat_upper, dtype=float) - flat_initial) / flat_scales

    def full_values(scaled_values):
        values = dict(fixed_values)
        values.update(
            unravel(
                jnp.asarray(flat_initial)
                + jnp.asarray(flat_scales) * scaled_values
            )
        )
        return values

    def residuals(scaled_values):
        return model.production_residuals(full_values(scaled_values))

    residual_jit = jax.jit(residuals)
    jacobian_jit = jax.jit(jax.jacfwd(residuals))

    def scipy_residual(scaled_values):
        return np.asarray(residual_jit(jnp.asarray(scaled_values)), dtype=float)

    def scipy_jacobian(scaled_values):
        return np.asarray(jacobian_jit(jnp.asarray(scaled_values)), dtype=float)

    initial_scaled = np.zeros_like(flat_initial)
    initial_residual = scipy_residual(initial_scaled)
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        bounds=(scaled_lower, scaled_upper),
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=production.PRODUCTION_MAP_STEPS,
        verbose=2,
    )
    final_scaled = np.asarray(result.x, dtype=float)
    optimized = full_values(final_scaled)
    final_residual = scipy_residual(final_scaled)
    jacobian = scipy_jacobian(final_scaled)
    scaled_gradient = jacobian.T @ final_residual
    return optimized, {
        "optimizer": "scipy least_squares reduced optical fit",
        "converged": bool(result.success),
        "strict_gradient_converged": bool(np.max(np.abs(scaled_gradient)) <= 1.0e-9),
        "status": int(result.status),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "initial_residual_cost": float(0.5 * np.sum(initial_residual**2)),
        "residual_cost": float(0.5 * np.sum(final_residual**2)),
        "scaled_gradient_norm": float(np.max(np.abs(scaled_gradient))),
    }


def _metrics(data, prediction, values):
    counts = np.asarray(data["counts"], dtype=float)
    observed = np.asarray(data["sums"], dtype=float) / counts
    prediction = np.asarray(prediction, dtype=float).reshape(counts.shape)
    post_mask = np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    offset = _scalar(values, "post_reset_tile_log_offset")
    observed = observed + offset * post_mask[None, :]
    residual = observed - prediction
    likelihood_weight = production.default_likelihood_weight(counts)
    scaled_data_residual = (
        np.sqrt(likelihood_weight / int(data["n_tiles"]))
        / production.LOG_NOISE_SIGMA
        * np.sqrt(counts)
        * (-residual)
    )
    nonreference = np.ones(residual.shape[1], dtype=bool)
    nonreference[np.asarray(data["first_by_led"], dtype=int)] = False
    return {
        "group_rms_log": float(np.sqrt(np.mean(residual**2))),
        "count_weighted_rms_log": float(
            np.sqrt(np.sum(counts * residual**2) / np.sum(counts))
        ),
        "nonreference_group_rms_log": float(
            np.sqrt(np.mean(residual[:, nonreference] ** 2))
        ),
        "data_chi2": float(np.sum(scaled_data_residual**2)),
        "max_abs_residual_log": float(np.max(np.abs(residual))),
    }


def _parameter_rows(target_values, forced_values, source_archive, target_state, forced_state):
    rows = []
    def add(name, baseline, forced, source=None):
        rows.append(
            {
                "parameter": name,
                "wfi01_baseline": float(baseline),
                "wfi01_fixed_wfi10_rate": float(forced),
                "wfi10_source": "" if source is None else float(source),
                "delta_fixed_minus_baseline": float(forced - baseline),
            }
        )

    for layer in range(3):
        for component_index, component in enumerate(("thickness_nm", "n", "k")):
            name = f"ar_layer{layer + 1}_{component}"
            baseline = np.asarray(target_state["ar_parameters"]).reshape(3, 3)[layer, component_index]
            forced = np.asarray(forced_state["ar_parameters"]).reshape(3, 3)[layer, component_index]
            source = np.asarray(source_archive["ar_parameters"]).reshape(-1, 3, 3)[0, layer, component_index]
            if component == "thickness_nm":
                # Serialized AR thickness coordinates are in 100 nm units.
                baseline *= 100.0
                forced *= 100.0
                source *= 100.0
            add(name, baseline, forced, source)

    for name in (
        "absolute_hgcdte_n_log_scale",
        "absolute_hgcdte_k_log_scale",
        "ice_n_log_constant",
        "post_reset_tile_log_offset",
    ):
        source_name = name
        if name == "ice_n_log_constant":
            source_name = "ice_n_log_constant"
        add(
            name,
            _scalar(target_values, name),
            _scalar(forced_values, name),
            _scalar(source_archive, source_name) if source_name in source_archive else None,
        )

    for name in (
        "ar_n_log_modulation",
        "hgcdte_n_log_modulation",
        "hgcdte_k_log_modulation",
    ):
        target = np.asarray(target_values[name], dtype=float).reshape(-1)
        forced = np.asarray(forced_values[name], dtype=float).reshape(-1)
        source = np.asarray(source_archive[name], dtype=float).reshape(-1)
        for index, (baseline, fitted, source_value) in enumerate(zip(target, forced, source)):
            add(f"{name}[{index}]", baseline, fitted, source_value)
    return rows


def _write_group_csv(
    path,
    data,
    baseline_prediction,
    forced_prediction,
    baseline_values,
    forced_values,
):
    counts = np.asarray(data["counts"], dtype=float)
    observed = np.asarray(data["sums"], dtype=float) / counts
    post_mask = np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    baseline_observed = observed[0] + _scalar(
        baseline_values, "post_reset_tile_log_offset"
    ) * post_mask
    forced_observed = observed[0] + _scalar(
        forced_values, "post_reset_tile_log_offset"
    ) * post_mask
    baseline_prediction = np.asarray(baseline_prediction).reshape(-1)
    forced_prediction = np.asarray(forced_prediction).reshape(-1)
    with Path(path).open("w", newline="") as handle:
        fields = (
            "group", "days", "led", "repeat", "observed_adjusted_log",
            "baseline_model_log", "fixed_rate_model_log",
            "baseline_residual_log", "fixed_rate_residual_log",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, group in enumerate(data["groups"]):
            writer.writerow(
                {
                    "group": int(group),
                    "days": float(data["days"][index]),
                    "led": int(data["led"][index]),
                    "repeat": int(data["repeat"][index]),
                    "observed_adjusted_log": float(forced_observed[index]),
                    "baseline_model_log": float(baseline_prediction[index]),
                    "fixed_rate_model_log": float(forced_prediction[index]),
                    "baseline_residual_log": float(
                        baseline_observed[index] - baseline_prediction[index]
                    ),
                    "fixed_rate_residual_log": float(
                        forced_observed[index] - forced_prediction[index]
                    ),
                }
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-statistics", type=Path, required=True)
    parser.add_argument("--source-statistics", type=Path, required=True)
    parser.add_argument("--target-baseline", type=Path, required=True)
    parser.add_argument("--source-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-id", type=int, default=136)
    parser.add_argument("--target-detector", default="WFI01")
    parser.add_argument("--source-detector", default="WFI10")
    parser.add_argument(
        "--fix-source-thickness",
        action="store_true",
        help=(
            "Also fix WFI01's pre/post-reset initial thicknesses to the WFI10 MAP; "
            "by default only the WFI10 rate-node trajectory is fixed."
        ),
    )
    parser.add_argument(
        "--fix-source-optics",
        action="store_true",
        help=(
            "Also fix WFI01's AR, ice-index, and HgCdTe optical parameters to "
            "the WFI10 MAP values."
        ),
    )
    parser.add_argument(
        "--fix-ice-n",
        action="store_true",
        help="Fix WFI01's constant ice real index at the production value 1.25.",
    )
    parser.add_argument("--ar-mean-prior-sigma", type=float, default=10.0)
    parser.add_argument(
        "--absolute-hgcdte-n-scale-prior-sigma-log",
        type=float,
        default=PRODUCTION_ABSOLUTE_HGCDTE_N_SCALE_PRIOR_SIGMA_LOG,
    )
    parser.add_argument(
        "--ar-hgcdte-modulation-prior-sigma-log",
        type=float,
        default=PRODUCTION_AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    )
    parser.add_argument(
        "--hgcdte-n-modulation-prior-sigma-log",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ice-n-constant-prior-sigma-log",
        type=float,
        default=PRODUCTION_ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.tile_id < 0:
        raise ValueError("--tile-id must be non-negative")
    target_detector = args.target_detector.upper()
    source_detector = args.source_detector.upper()
    target_baseline_archive_path = args.target_baseline / "posterior_samples.npz"
    source_baseline_archive_path = args.source_baseline / "posterior_samples.npz"
    if not target_baseline_archive_path.is_file():
        raise FileNotFoundError(target_baseline_archive_path)
    if not source_baseline_archive_path.is_file():
        raise FileNotFoundError(source_baseline_archive_path)

    target_data = production.load_tile_statistics(args.target_statistics)
    source_data = production.load_tile_statistics(args.source_statistics)
    _check_matching_observations(target_data, source_data)
    target_archive = _load_archive(target_baseline_archive_path)
    source_archive = _load_archive(source_baseline_archive_path)
    source_rates = _source_rate_nodes(source_archive, target_data)
    target_fingerprint = production.statistics_fingerprint(target_data)
    if str(np.asarray(target_archive["statistics_fingerprint"]).item()) != target_fingerprint:
        raise ValueError("Target baseline archive does not match target statistics")
    if str(np.asarray(source_archive["statistics_fingerprint"]).item()) != production.statistics_fingerprint(source_data):
        raise ValueError("Source baseline archive does not match source statistics")

    context = production.load_response_context(
        target_data["sca_names"], sed_weighting="energy", qe_response_mode="none"
    )
    target_data["qe_columns"] = context["qe_columns"]
    target_data["response_wavelengths_nm"] = context["wavelengths"]
    target_data["response_wavelength_count"] = int(context["wavelengths"].size)
    rate_reference = np.full(
        (target_data["n_scas"], target_data["n_tiles"], len(target_data["node_days"])),
        np.log(PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
    )
    model = production.build_model(
        target_data,
        context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=rate_reference,
        entropy_weight=0.0,
        rate_prior_log_mean=rate_reference,
        trajectory_rate_prior_sigma_log=PRODUCTION_TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            args.absolute_hgcdte_n_scale_prior_sigma_log
        ),
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        ice_n_constant_prior_sigma_log=args.ice_n_constant_prior_sigma_log,
        fit_ar_hgcdte_modulation=True,
        ar_hgcdte_modulation_prior_sigma_log=args.ar_hgcdte_modulation_prior_sigma_log,
        hgcdte_n_modulation_prior_sigma_log=args.hgcdte_n_modulation_prior_sigma_log,
    )
    initial_values = production.initial_values_for_data(
        target_data,
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        fit_ar_hgcdte_modulation=True,
    )
    initial_values = production.load_map_warm_start(
        target_baseline_archive_path,
        initial_values,
        target_data,
    )
    initial_values["np_log_rate_nodes"] = np.log(source_rates).reshape(
        np.asarray(initial_values["np_log_rate_nodes"]).shape
    )
    fixed_site_names = set()
    if args.fix_source_thickness:
        for name in ("initial_thickness_log_nm", "reset_initial_thickness_log_nm"):
            initial_values[name] = np.asarray(source_archive[name], dtype=float).reshape(
                np.asarray(initial_values[name]).shape
            )
        fixed_site_names.update(
            ("initial_thickness_log_nm", "reset_initial_thickness_log_nm")
        )
    if args.fix_source_optics:
        for name in SOURCE_OPTICAL_SITES:
            if name not in source_archive:
                raise ValueError(f"Source archive is missing optical site {name}")
            initial_values[name] = np.asarray(source_archive[name], dtype=float).reshape(
                np.asarray(initial_values[name]).shape
            )
        fixed_site_names.update(SOURCE_OPTICAL_SITES)
    if args.fix_ice_n:
        initial_values["ice_n_log_constant"] = np.zeros_like(
            initial_values["ice_n_log_constant"]
        )
        fixed_site_names.add("ice_n_log_constant")
    variable_sites = tuple(
        name for name in VARIABLE_SITES if name not in fixed_site_names
    )

    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    forced_values, diagnostics = _reduced_least_squares(
        model,
        initial_values,
        variable_sites,
        jax,
        jnp,
        ar_mean_prior_sigma=args.ar_mean_prior_sigma,
        absolute_hgcdte_n_scale_prior_sigma_log=(
            args.absolute_hgcdte_n_scale_prior_sigma_log
        ),
        ar_hgcdte_modulation_prior_sigma_log=args.ar_hgcdte_modulation_prior_sigma_log,
        hgcdte_n_modulation_prior_sigma_log=(
            args.hgcdte_n_modulation_prior_sigma_log
            if args.hgcdte_n_modulation_prior_sigma_log is not None
            else args.ar_hgcdte_modulation_prior_sigma_log
        ),
        ice_n_constant_prior_sigma_log=args.ice_n_constant_prior_sigma_log,
    )
    forced_state = model.production_state(forced_values)
    forced_prediction = np.asarray(forced_state["prediction"], dtype=float)
    target_state = {
        "ar_parameters": target_archive["ar_parameters"],
        "prediction": target_archive["_map_least_squares_prediction"],
    }
    baseline_prediction = np.asarray(target_state["prediction"], dtype=float)
    baseline_values = {
        name: np.asarray(target_archive[name])[0]
        for name in VARIABLE_SITES
        if name in target_archive
    }
    baseline_metrics = _metrics(target_data, baseline_prediction, baseline_values)
    forced_metrics = _metrics(target_data, forced_prediction, forced_values)
    source_metrics = _metrics(
        source_data,
        np.asarray(source_archive["_map_least_squares_prediction"], dtype=float),
        {
            name: np.asarray(source_archive[name])[0]
            for name in VARIABLE_SITES
            if name in source_archive
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_archive_path = args.output_dir / "fixed_rate_fit.npz"
    np.savez_compressed(
        fixed_archive_path,
        **{name: np.asarray(value) for name, value in forced_values.items()},
        **{name: np.asarray(value) for name, value in forced_state.items()},
        source_node_days=np.asarray(target_data["node_days"], dtype=float),
        source_node_rates_nm_day=source_rates,
        target_statistics_fingerprint=np.asarray(target_fingerprint),
        source_statistics_fingerprint=np.asarray(production.statistics_fingerprint(source_data)),
    )

    group_csv = args.output_dir / "fixed_rate_group_predictions.csv"
    _write_group_csv(
        group_csv,
        target_data,
        baseline_prediction,
        forced_prediction,
        baseline_values,
        forced_values,
    )
    parameter_rows = _parameter_rows(
        baseline_values,
        forced_values,
        source_archive,
        target_state,
        forced_state,
    )
    parameter_csv = args.output_dir / "parameter_comparison.csv"
    with parameter_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=parameter_rows[0].keys())
        writer.writeheader()
        writer.writerows(parameter_rows)

    payload = {
        "diagnostic": "wfi01_tile136_fixed_wfi10_tile136_rate_trajectory",
        "target_detector": target_detector,
        "source_detector": source_detector,
        "tile_id": int(args.tile_id),
        "tile_size": int(args.tile_size),
        "target_statistics": str(args.target_statistics.resolve()),
        "source_statistics": str(args.source_statistics.resolve()),
        "target_baseline": str(target_baseline_archive_path.resolve()),
        "source_baseline": str(source_baseline_archive_path.resolve()),
        "fixed_trajectory_definition": (
            "WFI10 tile-136 MAP node_instantaneous_rate_nm_day; "
            + (
                "WFI10 MAP pre/post-reset initial thicknesses are also fixed"
                if args.fix_source_thickness
                else "WFI01 initial and post-reset thickness baselines remain free"
            )
        ),
        "fixed_optical_definition": (
            "WFI10 MAP AR, ice-index, and HgCdTe optical sites are also fixed"
            if args.fix_source_optics
            else "WFI01 optical sites remain free"
        ),
        "fixed_ice_n_definition": (
            "WFI01 constant ice n fixed at 1.25"
            if args.fix_ice_n
            else "WFI01 constant ice n remains free"
        ),
        "variable_sites": list(variable_sites),
        "node_days": np.asarray(target_data["node_days"], dtype=float).tolist(),
        "fixed_source_node_rates_nm_day": source_rates.tolist(),
        "prior_settings": {
            "ar_mean_prior_sigma": float(args.ar_mean_prior_sigma),
            "absolute_hgcdte_n_scale_prior_sigma_log": float(
                args.absolute_hgcdte_n_scale_prior_sigma_log
            ),
            "ar_hgcdte_modulation_prior_sigma_log": float(
                args.ar_hgcdte_modulation_prior_sigma_log
            ),
            "hgcdte_n_modulation_prior_sigma_log": float(
                args.hgcdte_n_modulation_prior_sigma_log
                if args.hgcdte_n_modulation_prior_sigma_log is not None
                else args.ar_hgcdte_modulation_prior_sigma_log
            ),
            "ice_n_constant_prior_sigma_log": float(args.ice_n_constant_prior_sigma_log),
            "trajectory_rate_prior_mean_nm_day": PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
            "trajectory_rate_prior_sigma_log": PRODUCTION_TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
            "fit_incidence_angle": False,
            "fit_ice_n_constant": True,
            "fit_ar_hgcdte_modulation": True,
        },
        "baseline": {
            "map_objective": _summary_objective(args.target_baseline / "map_summary.md"),
            "metrics": baseline_metrics,
            "source_wfi10_metrics": source_metrics,
        },
        "forced_fit": {
            "metrics": forced_metrics,
            "optimizer": diagnostics,
        },
        "outputs": {
            "archive": str(fixed_archive_path.resolve()),
            "groups": str(group_csv.resolve()),
            "parameters": str(parameter_csv.resolve()),
        },
    }
    json_path = args.output_dir / "fixed_rate_fit.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {fixed_archive_path}")
    print(f"Wrote {group_csv}")
    print(f"Wrote {parameter_csv}")
    print(
        "Target baseline weighted RMS: "
        f"{baseline_metrics['count_weighted_rms_log']:.9g} log"
    )
    print(
        "Fixed-rate weighted RMS: "
        f"{forced_metrics['count_weighted_rms_log']:.9g} log"
    )
    print(
        "Fixed WFI10 node-rate range: "
        f"{np.min(source_rates):.9g} to {np.max(source_rates):.9g} nm/day"
    )


if __name__ == "__main__":
    main()
