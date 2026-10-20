"""Fit one tile's ice trajectory using a saved relative optical response."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

import fit_wfi_tmm_production as production
from wfi_relative_response import (
    PRODUCTION_ENTROPY_WEIGHT,
    PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
    context_fingerprint,
    build_ice_only_evaluator,
    fit_ice_only,
    fit_ice_only_batch,
    fit_ice_only_reusable,
    load_response_cache,
    load_warm_values,
    relative_state,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--response-cache", type=Path, required=True)
    parser.add_argument("--initial-fit", type=Path, required=True)
    parser.add_argument("--initial-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def statistics_fingerprint(statistics_path):
    """Fingerprint the sufficient statistics used by a cached fit."""
    digest = hashlib.sha256()
    with Path(statistics_path).open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["group"]))
    if not rows:
        raise ValueError(f"No statistics rows found in {statistics_path}")
    for field in ("n_pair_valid", "sum_log_change", "sum_squared_log_change"):
        if field not in rows[0]:
            raise ValueError(f"Missing {field} in {statistics_path}")
        digest.update(
            np.asarray([float(row[field]) for row in rows], dtype=np.float64).tobytes()
        )
    return digest.hexdigest()


def write_outputs(output_dir, data, values, state, diagnostics, cache_path, jax, jnp):
    output_dir.mkdir(parents=True, exist_ok=True)
    observed = np.asarray(data["sums"], dtype=float) / np.asarray(data["counts"], dtype=float)
    observed += np.asarray(values["post_reset_tile_log_offset"]).reshape(-1, 1) * (
        np.asarray(data["days"])[None, :] >= float(data["reset_day"])
    )
    prediction = np.asarray(state["prediction"], dtype=float)
    residual = observed - prediction
    with (output_dir / "tile_group_predictions.csv").open("w", newline="") as handle:
        fields = [
            "detector",
            "tile_id",
            "group",
            "days",
            "led",
            "repeat",
            "observed_log_change",
            "model_log_change",
            "residual_log",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, group in enumerate(data["groups"]):
            writer.writerow(
                {
                    "detector": str(data["sca_names"][0]),
                    "tile_id": int(data["tile_ids"][0]),
                    "group": int(group),
                    "days": float(data["days"][index]),
                    "led": int(data["led"][index]),
                    "repeat": int(data["repeat"][index]),
                    "observed_log_change": float(observed[0, index]),
                    "model_log_change": float(prediction[0, index]),
                    "residual_log": float(residual[0, index]),
                }
            )

    rates = np.asarray(state["rates"], dtype=float).reshape(-1)
    node_rates = np.asarray(state["node_rates"], dtype=float).reshape(-1)
    node_ice = np.asarray(state["node_ice"], dtype=float).reshape(-1)
    ice = np.asarray(state["ice"], dtype=float).reshape(-1)
    days = np.asarray(data["days"], dtype=float)
    ar_parameters = np.asarray(
        production._physical_ar_parameters(
            jnp.asarray(np.asarray(values["ar_tile_raw"], dtype=float)), jax, jnp
        ),
        dtype=float,
    ).reshape(-1, production.AR_LAYERS, 3)[0]
    tile_id = int(np.asarray(data["tile_ids"]).reshape(-1)[0])
    detector = str(np.asarray(data["sca_names"]).reshape(-1)[0])
    tile_metadata = data["metadata"][tile_id]
    node_multipliers = np.ones_like(node_rates)
    np.savez_compressed(
        output_dir / "posterior_samples.npz",
        map_only=np.asarray(1),
        tile_ids=np.asarray(data["tile_ids"], dtype=int),
        sca_names=np.asarray(data["sca_names"]).astype(str),
        observation_days=days,
        observation_led=np.asarray(data["led"], dtype=int),
        observation_repeat=np.asarray(data["repeat"], dtype=int),
        np_log_rate_nodes=np.asarray(values["np_log_rate_nodes"]),
        np_node_multipliers=node_multipliers[None, None, :],
        initial_thickness_log_nm=np.asarray(values["initial_thickness_log_nm"]),
        reset_initial_thickness_log_nm=np.asarray(values["reset_initial_thickness_log_nm"]),
        post_reset_tile_log_offset=np.asarray(values["post_reset_tile_log_offset"]),
        ar_tile_raw=np.asarray(values["ar_tile_raw"]),
        absolute_hgcdte_n_log_scale=np.asarray(values["absolute_hgcdte_n_log_scale"]),
        absolute_hgcdte_k_log_scale=np.asarray(values["absolute_hgcdte_k_log_scale"]),
        incidence_angle_deg=np.asarray(values["incidence_angle_deg"]),
        tile_ice_thickness=ice[None, None, :],
        tile_instantaneous_rate_nm_day=rates[None, None, :],
        np_node_thickness=node_ice[None, None, :],
        node_instantaneous_rate_nm_day=node_rates[None, None, :],
        response_cache=np.asarray(str(cache_path.resolve())),
        fit_mode=np.asarray("relative_response_production"),
        statistics_fingerprint=np.asarray(statistics_fingerprint(data["statistics_path"])),
        statistics_path=np.asarray(str(Path(data["statistics_path"]).resolve())),
        observation_program=np.asarray(data["program"], dtype=int),
        groups=np.asarray(data["groups"], dtype=int),
        ice_reset_day=np.asarray(float(data["reset_day"])),
        fit_incidence_angle=np.asarray(0),
        fully_np_trajectory=np.asarray(1),
        map_objective_history=np.asarray(
            [diagnostics["initial_residual_cost"], diagnostics["residual_cost"]],
            dtype=float,
        ),
        trajectory_rate_prior_mean_nm_day=np.asarray(
            PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY
        ),
        trajectory_rate_initialization_nm_day=np.asarray(
            diagnostics["trajectory_rate_initialization_nm_day"]
        ),
        trajectory_rate_prior_sigma_log=np.asarray(
            production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG
        ),
        trajectory_entropy_weight=np.asarray(PRODUCTION_ENTROPY_WEIGHT),
        trajectory_regularization=np.asarray("none"),
        map_optimizer=np.asarray("least-squares"),
        map_steps=np.asarray(production.PRODUCTION_MAP_STEPS),
        map_parameter_scaling=np.asarray("prior"),
        ar_parameters=ar_parameters.reshape(1, 1, 1, -1),
    )

    ar_fields = ["detector", "tile_id", "tile_row", "tile_col"]
    for layer in range(production.AR_LAYERS):
        for component in ("thickness", "n", "k"):
            for suffix in ("p16", "median", "p84"):
                ar_fields.append(f"ar_layer{layer + 1}_{component}_{suffix}")
    ar_row = {
        "detector": detector,
        "tile_id": tile_id,
        "tile_row": tile_metadata["tile_row"],
        "tile_col": tile_metadata["tile_col"],
    }
    for layer in range(production.AR_LAYERS):
        for component_index, component in enumerate(("thickness", "n", "k")):
            value = float(ar_parameters[layer, component_index])
            for suffix in ("p16", "median", "p84"):
                ar_row[f"ar_layer{layer + 1}_{component}_{suffix}"] = value
    with (output_dir / "tile_ar_posterior.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ar_fields)
        writer.writeheader()
        writer.writerow(ar_row)

    valid = np.ones(residual.shape[1], dtype=bool)
    valid[np.asarray(data["first_by_led"], dtype=int)] = False
    residual_values = residual[0, valid]
    node_days = np.asarray(data["node_days"], dtype=float)
    reset_day = float(data["reset_day"])
    pre_indices = np.flatnonzero(node_days < reset_day)
    post_indices = np.flatnonzero(node_days >= reset_day)
    pre_accumulation = node_ice[pre_indices[-1]] - node_ice[pre_indices[0]]
    post_accumulation = node_ice[post_indices[-1]] - node_ice[post_indices[0]]
    pre_duration = reset_day - float(np.min(days))
    post_duration = float(np.max(days)) - reset_day
    mean_rate = float(
        (pre_accumulation + post_accumulation) / (pre_duration + post_duration)
    )
    initial_rate = float(rates[0])
    final_rate = float(rates[-1])
    rate_row = {
        "detector": detector,
        "tile_id": tile_id,
        "tile_row": tile_metadata["tile_row"],
        "tile_col": tile_metadata["tile_col"],
        "log_rate_offset_p16": 0.0,
        "log_rate_offset_median": 0.0,
        "log_rate_offset_p84": 0.0,
        "initial_rate_nm_day_p16": initial_rate,
        "initial_rate_nm_day_median": initial_rate,
        "initial_rate_nm_day_p84": initial_rate,
        "final_rate_nm_day_p16": final_rate,
        "final_rate_nm_day_median": final_rate,
        "final_rate_nm_day_p84": final_rate,
        "mean_rate_nm_day_p16": mean_rate,
        "mean_rate_nm_day_median": mean_rate,
        "mean_rate_nm_day_p84": mean_rate,
    }
    with (output_dir / "tile_rate_posterior.csv").open("w", newline="") as handle:
        fields = [
            "detector",
            "tile_id",
            "tile_row",
            "tile_col",
            "log_rate_offset_p16",
            "log_rate_offset_median",
            "log_rate_offset_p84",
            "initial_rate_nm_day_p16",
            "initial_rate_nm_day_median",
            "initial_rate_nm_day_p84",
            "final_rate_nm_day_p16",
            "final_rate_nm_day_median",
            "final_rate_nm_day_p84",
            "mean_rate_nm_day_p16",
            "mean_rate_nm_day_median",
            "mean_rate_nm_day_p84",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(rate_row)

    summary = "\n".join(
        [
            "# Relative-response production fit",
            "",
            f"- Statistics: `{Path(data['statistics_path']).resolve() if 'statistics_path' in data else 'input statistics'}`.",
            f"- Response cache: `{cache_path.resolve()}`.",
            f"- Evaluations: `{diagnostics['nfev']}`.",
            f"- Converged: `{diagnostics['converged']}`.",
            f"- Runtime: `{diagnostics['total_seconds']:.6g}` seconds.",
            f"- Free parameters: `{diagnostics['parameter_count']}`.",
            f"- Residual RMS excluding LED reference groups: `{np.sqrt(np.mean(residual_values**2)):.12g}` log.",
            f"- Maximum absolute residual excluding LED reference groups: `{np.max(np.abs(residual_values)):.12g}` log.",
            f"- Warm-start rate: `{diagnostics['trajectory_rate_initialization_nm_day']:.12g}` nm/day.",
            f"- Final rate: `{rates[-1]:.12g}` nm/day.",
            f"- Mean rate: `{mean_rate:.12g}` nm/day.",
            f"- Trajectory prior: `{PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY}` nm/day, log sigma `{production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG}`.",
            f"- Trajectory entropy weight: `{PRODUCTION_ENTROPY_WEIGHT}`.",
            "",
        ]
    )
    (output_dir / "map_summary.md").write_text(summary)


def fit_one(
    statistics_path,
    response_cache_path,
    initial_fit,
    initial_statistics,
    output_dir,
    trajectory_rate_initialization_nm_day=PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
):
    """Fit one tile from a cached response and a previous tile fit."""
    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    statistics_path = Path(statistics_path)
    response_cache_path = Path(response_cache_path)
    data = production.load_tile_statistics(statistics_path)
    data["statistics_path"] = str(statistics_path)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    cache = load_response_cache(response_cache_path, data, context)
    values = load_warm_values(
        data,
        initial_fit,
        initial_statistics,
        trajectory_rate_initialization_nm_day=trajectory_rate_initialization_nm_day,
    )
    values, diagnostics = fit_ice_only(data, cache, values, jax, jnp)
    state = relative_state(data, values, cache, jax, jnp)
    write_outputs(
        output_dir,
        data,
        values,
        state,
        diagnostics,
        response_cache_path,
        jax,
        jnp,
    )
    print(f"Wrote relative-response fit under {output_dir}", flush=True)
    print(
        f"Runtime {diagnostics['total_seconds']:.3f}s; "
        f"{diagnostics['parameter_count']} parameters; "
        f"{diagnostics['nfev']} evaluations",
        flush=True,
    )
    return diagnostics


def main():
    args = parse_args()
    fit_one(
        args.statistics,
        args.response_cache,
        args.initial_fit,
        args.initial_statistics,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
