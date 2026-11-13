"""Run a reduced WFI01 fixed-ice-index multi-start MAP search.

The search varies AR thickness coordinates, the two ice thickness starts, the
non-parametric rate nodes, and the post-reset offset.  WFI01's real-index
reference state and other optical nuisance values remain fixed.  AR n/k
coordinates are technically free but receive a very tight prior around the
WFI01 state so the search remains a thickness/rate basin diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production


TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY = 12.0
TRAJECTORY_RATE_PRIOR_SIGMA_LOG = 1.5
AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG = 0.10
ICE_N_CONSTANT_PRIOR_SIGMA_LOG = 0.01
FREE_SITES = {
    "ar_tile_raw",
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "post_reset_tile_log_offset",
    "np_log_rate_nodes",
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _archive_path(path):
    path = Path(path)
    return path / "posterior_samples.npz" if path.is_dir() else path


def _load_context(statistics_path):
    data = production.load_tile_statistics(statistics_path)
    context = production.load_response_context(
        data["sca_names"],
        sed_weighting="energy",
        qe_response_mode="none",
    )
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    return data, context


def _load_values(statistics_path, map_path, data):
    initial = production.initial_values_for_data(
        data,
        constant_rate=False,
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        fit_ar_hgcdte_modulation=True,
    )
    archive_path = _archive_path(map_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        transferred = dict(initial)
        for name, target in initial.items():
            if name not in archive:
                continue
            value = np.asarray(archive[name], dtype=float)
            target_shape = np.asarray(target).shape
            while value.ndim > len(target_shape) and value.shape[0] == 1:
                value = value[0]
            if value.shape != target_shape:
                raise ValueError(
                    f"Initialization archive shape mismatch for {name}: "
                    f"{value.shape} != {target_shape}"
                )
            transferred[name] = value.copy()
    return transferred


def _ar_base_n(map_path):
    with np.load(_archive_path(map_path), allow_pickle=False) as archive:
        parameters = np.asarray(archive["ar_parameters"], dtype=float).reshape(-1, 3, 3)[0]
    return parameters[:, 1].copy()


def _seed_values(baseline, source, *, high_rate=None):
    values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in baseline.items()
    }
    source = {
        name: np.asarray(value, dtype=float)
        for name, value in source.items()
    }
    for name in (
        "initial_thickness_log_nm",
        "reset_initial_thickness_log_nm",
        "post_reset_tile_log_offset",
        "np_log_rate_nodes",
    ):
        values[name] = source[name].copy()
    values["ar_tile_raw"] = baseline["ar_tile_raw"].copy()
    values["ar_tile_raw"][..., ::2] = source["ar_tile_raw"][..., ::2]
    values["ice_n_log_constant"] = np.zeros_like(values["ice_n_log_constant"])
    if high_rate is not None:
        values["np_log_rate_nodes"] = np.full_like(
            values["np_log_rate_nodes"], np.log(float(high_rate))
        )
    return values


def _rate_summary(data, values, state, jax, jnp):
    minimum = float(np.min(data["days"]))
    maximum = float(np.max(data["days"]))
    reset = float(data["reset_day"])
    node_days = np.asarray(data["node_days"], dtype=float)
    integration_times = np.unique(
        np.concatenate(([minimum], node_days, [maximum, reset]))
    )
    profile = np.asarray(values["np_log_rate_nodes"], dtype=float).reshape(-1, node_days.size)[0]
    integrated = np.asarray(
        production.integrate_rate_trajectory(
            jax,
            jnp,
            jnp.asarray([reset, maximum]),
            jnp.asarray(node_days),
            jnp.asarray(integration_times),
            jnp.asarray(profile),
        )[0],
        dtype=float,
    )
    pre = integrated[0] / (reset - minimum)
    post = (integrated[1] - integrated[0]) / (maximum - reset)
    overall = (integrated[1]) / (maximum - minimum)
    rates = np.asarray(state["rates"], dtype=float).reshape(-1)
    return {
        "pre_reset_mean_rate_nm_day": float(pre),
        "post_reset_mean_rate_nm_day": float(post),
        "reset_aware_mean_rate_nm_day": float(overall),
        "initial_node_rate_nm_day": float(rates[0]),
        "final_node_rate_nm_day": float(rates[-1]),
    }


def _metrics(data, model, values, jax, jnp):
    residual = np.asarray(model.production_residuals(values), dtype=float)
    state = model.production_state(values)
    prediction = np.asarray(state["prediction"], dtype=float).reshape(-1)
    counts = np.asarray(data["counts"], dtype=float).reshape(-1)
    observed = np.asarray(data["sums"], dtype=float).reshape(-1) / np.maximum(counts, 1.0)
    post = np.asarray(data["days"], dtype=float) >= float(data["reset_day"])
    observed = observed + float(np.asarray(values["post_reset_tile_log_offset"]).reshape(-1)[0]) * post
    data_residual = prediction - observed
    valid = np.ones(data_residual.size, dtype=bool)
    valid[np.asarray(data["first_by_led"], dtype=int)] = False
    valid_residual = data_residual[valid]
    valid_counts = counts[valid]
    ar = np.asarray(state["ar_parameters"], dtype=float).reshape(-1, 3, 3)[0]
    result = {
        "residual_objective": float(0.5 * np.sum(residual**2)),
        "data_rms_log": float(np.sqrt(np.mean(valid_residual**2))),
        "data_count_weighted_rms_log": float(
            np.sqrt(np.sum(valid_counts * valid_residual**2) / np.sum(valid_counts))
        ),
        "initial_ice_thickness_nm": float(
            np.exp(np.asarray(values["initial_thickness_log_nm"]).reshape(-1)[0])
        ),
        "reset_initial_ice_thickness_nm": float(
            np.exp(np.asarray(values["reset_initial_thickness_log_nm"]).reshape(-1)[0])
        ),
        "ar_thickness_nm": (ar[:, 0] * 100.0).tolist(),
        "ar_n": ar[:, 1].tolist(),
        "ar_k": ar[:, 2].tolist(),
    }
    result.update(_rate_summary(data, values, state, jax, jnp))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        action="append",
        nargs=2,
        metavar=("NAME", "ARCHIVE"),
        required=True,
        help="Seed name and deterministic MAP archive/directory; repeatable.",
    )
    parser.add_argument(
        "--high-rate",
        action="append",
        nargs=2,
        metavar=("NAME", "RATE"),
        default=[],
        help="Additional baseline seed with a constant initial rate; repeatable.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data, context = _load_context(args.statistics)
    baseline = _load_values(args.statistics, args.baseline, data)
    ar_base_n = _ar_base_n(args.baseline)
    rate_prior = np.full(
        np.asarray(baseline["np_log_rate_nodes"]).shape,
        np.log(TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
    )
    ar_center = np.asarray(baseline["ar_tile_raw"], dtype=float).reshape(-1)
    ar_sigma = np.full(ar_center.size, 1.0e-6, dtype=float)
    ar_sigma[::2] = 10.0
    model = production.build_model(
        data,
        context,
        smoothness_weight=0.0,
        entropy_reference_log_rates=rate_prior,
        entropy_weight=0.0,
        rate_prior_log_mean=rate_prior,
        trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        absolute_hgcdte_n_scale_prior_sigma_log=0.01,
        fit_incidence_angle=False,
        fit_ice_n_constant=True,
        ice_n_constant_prior_sigma_log=ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
        fixed_ar_base_n=ar_base_n,
        ar_prior_center_raw=ar_center,
        ar_prior_sigma_raw=ar_sigma,
        fit_ar_hgcdte_modulation=True,
        ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
        hgcdte_n_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
    )
    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    sources = [(name, Path(path), None) for name, path in args.seed]
    sources.extend(
        (name, args.baseline, float(rate)) for name, rate in args.high_rate
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, source_path, high_rate in sources:
        source = _load_values(args.statistics, source_path, data)
        initial = _seed_values(baseline, source, high_rate=high_rate)
        fixed = {
            key: value.copy()
            for key, value in initial.items()
            if key not in FREE_SITES
        }
        optimized, diagnostics = production.run_least_squares_map_optimization(
            model,
            initial,
            jax,
            trajectory_rate_prior_sigma_log=TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
            initial_thickness_prior_sigma_nm=5.0,
            absolute_hgcdte_n_scale_prior_sigma_log=0.01,
            ice_n_constant_prior_sigma_log=ICE_N_CONSTANT_PRIOR_SIGMA_LOG,
            ar_prior_sigma_raw=ar_sigma,
            ar_hgcdte_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
            hgcdte_n_modulation_prior_sigma_log=AR_HGCDTE_MODULATION_PRIOR_SIGMA_LOG,
            fixed_parameter_values=fixed,
        )
        metrics = _metrics(data, model, optimized, jax, jnp)
        seed_result = {
            "name": name,
            "source": str(source_path.resolve()),
            "high_rate_nm_day": high_rate,
            "free_sites": sorted(FREE_SITES),
            "diagnostics": diagnostics,
            "metrics": metrics,
        }
        results.append(seed_result)
        seed_dir = args.output_dir / name
        seed_dir.mkdir(parents=True, exist_ok=True)
        state = model.production_state(optimized)
        np.savez_compressed(
            seed_dir / "reduced_map.npz",
            **{key: np.asarray(value) for key, value in optimized.items()},
            **{key: np.asarray(value) for key, value in state.items()},
        )
        (seed_dir / "summary.json").write_text(
            json.dumps(seed_result, indent=2, sort_keys=True, default=_json_default) + "\n"
        )
        print(
            f"{name}: objective={diagnostics['objective']:.9g}; "
            f"data_rms={metrics['data_rms_log']:.9g}; "
            f"rate={metrics['reset_aware_mean_rate_nm_day']:.9g} nm/day"
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps({"results": results}, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


if __name__ == "__main__":
    main()
