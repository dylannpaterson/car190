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
    build_ice_only_constant_rate_evaluator,
    fit_ice_only,
    fit_ice_only_batch,
    fit_ice_only_constant_rate_reusable,
    fit_ice_only_reusable,
    load_response_cache,
    load_warm_values,
    portable_path_string,
    resolve_portable_path,
    relative_state,
)


DEFAULT_WARMUP_THICKNESS_PRIOR_LOG_SIGMA = 0.10
DEFAULT_WARMUP_THICKNESS_PRIOR_STAGES = "third-decon"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--response-cache", type=Path, required=True)
    parser.add_argument("--initial-fit", type=Path, required=True)
    parser.add_argument("--initial-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--third-decon-day", type=float, default=None)
    parser.add_argument(
        "--third-decon-initial-thickness-start-nm",
        type=float,
        default=None,
        help="Override the post-third-decon thickness warm start in nm.",
    )
    parser.add_argument("--fit-led-log-offsets", action="store_true")
    parser.add_argument(
        "--led-log-offset-prior-sigma-log",
        type=float,
        default=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    )
    parser.add_argument(
        "--warmup-prior-root",
        type=Path,
        default=None,
        help="Constant-rate warmup directory containing fits/tile_### archives.",
    )
    parser.add_argument("--thickness-prior-log-sigma", type=float, default=None)
    parser.add_argument(
        "--thickness-prior-stages",
        choices=("all", "third-decon"),
        default=None,
    )
    return parser.parse_args()


def resolve_warmup_prior_settings(
    third_decon_day,
    warmup_prior_root,
    thickness_prior_log_sigma,
    thickness_prior_stages,
):
    """Resolve the warmup-prior defaults without choosing a reset boundary."""
    if warmup_prior_root is None:
        if thickness_prior_log_sigma is not None or thickness_prior_stages is not None:
            raise ValueError("Thickness prior settings require warmup_prior_root")
        return None, None, None
    if third_decon_day is None:
        raise ValueError(
            "Warmup-centered thickness priors require an explicit third_decon_day"
        )
    if thickness_prior_log_sigma is None:
        thickness_prior_log_sigma = DEFAULT_WARMUP_THICKNESS_PRIOR_LOG_SIGMA
    thickness_prior_log_sigma = float(thickness_prior_log_sigma)
    if not np.isfinite(thickness_prior_log_sigma) or thickness_prior_log_sigma <= 0.0:
        raise ValueError("Thickness prior log sigma must be positive and finite")
    if thickness_prior_stages is None:
        thickness_prior_stages = DEFAULT_WARMUP_THICKNESS_PRIOR_STAGES
    if thickness_prior_stages not in ("all", "third-decon") and not str(
        thickness_prior_stages
    ).startswith("reset-event:"):
        raise ValueError(
            "Thickness prior stages must be 'all', 'third-decon', or "
            "'reset-event:<event-name>'"
        )
    return (
        resolve_portable_path(warmup_prior_root),
        thickness_prior_log_sigma,
        thickness_prior_stages,
    )


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


def warmup_thickness_prior(
    warmup_root,
    tile_id,
    data,
    prior_sigma_log,
    prior_stages,
):
    """Load a per-tile thickness prior center from a constant-rate warmup."""
    if prior_stages not in ("all", "third-decon") and not str(
        prior_stages
    ).startswith("reset-event:"):
        raise ValueError(
            "prior_stages must be 'all', 'third-decon', or "
            "'reset-event:<event-name>'"
        )
    if not np.isfinite(prior_sigma_log) or prior_sigma_log <= 0.0:
        raise ValueError("prior_sigma_log must be positive and finite")
    archive_path = (
        Path(warmup_root)
        / "fits"
        / f"tile_{int(tile_id):03d}"
        / "constant_rate_map.npz"
    )
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    thickness_names, _ = production.reset_site_names(data)
    centers = {}
    sigmas = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        event_names = data["reset_event_names"]
        for index, name in enumerate(thickness_names):
            if index == 0:
                event_name = None
                event = None
            else:
                event_name = event_names[index - 1]
                configured_events = data.get("reset_events", ())
                event = (
                    configured_events[index - 1]
                    if len(configured_events) >= index
                    else None
                )
            use_warmup = prior_stages == "all"
            if prior_stages == "third-decon":
                use_warmup = event_name == "third_decon"
            elif str(prior_stages).startswith("reset-event:"):
                use_warmup = event_name == str(prior_stages).split(":", 1)[1]
            if not use_warmup:
                prior = {} if event is None else event.get("thickness_prior", {})
                centers[name] = np.asarray(
                    np.log(
                        float(
                            prior.get(
                                "median_nm",
                                production.INITIAL_THICKNESS_LOG_MEDIAN_NM,
                            )
                        )
                    )
                )
                sigmas[name] = float(
                    prior.get("log_sigma", production.INITIAL_THICKNESS_LOG_SIGMA)
                )
                continue
            if name not in archive:
                raise ValueError(f"Warmup archive lacks {name}: {archive_path}")
            centers[name] = np.asarray(archive[name], dtype=float).copy()
            sigmas[name] = prior_sigma_log
    return centers, sigmas


def write_outputs(
    output_dir,
    data,
    values,
    state,
    diagnostics,
    cache_path,
    jax,
    jnp,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    third_decon_initial_thickness_start_nm=None,
    constant_rate_warmup_diagnostics=None,
    fit_mode="relative_response_production",
    thickness_prior_log_sigma=None,
    thickness_prior_stages=None,
    warmup_prior_root=None,
    spline_knot_days=None,
    spline_knot_boundaries=None,
    spline_knot_segment_sizes=None,
    reset_event_initial_thickness_starts_nm=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    observed = np.asarray(data["sums"], dtype=float) / np.asarray(data["counts"], dtype=float)
    thickness_names, offset_names = production.reset_site_names(data)
    reset_days = production.reset_days_for_data(data)
    reset_masks = np.asarray(data["days"], dtype=float)[None, :] >= reset_days[:, None]
    reset_offsets = np.stack(
        [np.asarray(values[name]).reshape(-1) for name in offset_names], axis=0
    )
    observed += np.sum(reset_offsets[:, :, None] * reset_masks[:, None, :], axis=0)
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
    archive_values = dict(
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
        response_cache=np.asarray(portable_path_string(cache_path)),
        fit_mode=np.asarray(str(fit_mode)),
        statistics_fingerprint=np.asarray(statistics_fingerprint(data["statistics_path"])),
        statistics_path=np.asarray(portable_path_string(data["statistics_path"])),
        observation_program=np.asarray(data["program"], dtype=int),
        groups=np.asarray(data["groups"], dtype=int),
        ice_reset_day=np.asarray(float(reset_days[0])),
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
        fit_led_log_offsets=np.asarray(bool(fit_led_log_offsets)),
        led_log_offset_prior_sigma_log=np.asarray(
            float(led_log_offset_prior_sigma_log)
        ),
        constant_rate_map_warmup=np.asarray(
            int(constant_rate_warmup_diagnostics is not None)
        ),
    )
    if spline_knot_days is not None:
        archive_values["spline_knot_days"] = np.asarray(spline_knot_days, dtype=float)
    if spline_knot_boundaries is not None:
        archive_values["spline_knot_boundaries"] = np.asarray(
            spline_knot_boundaries, dtype=float
        )
    if spline_knot_segment_sizes is not None:
        archive_values["spline_knot_segment_sizes"] = np.asarray(
            spline_knot_segment_sizes, dtype=int
        )
    if thickness_prior_log_sigma is not None:
        archive_values["thickness_prior_log_sigma"] = np.asarray(
            float(thickness_prior_log_sigma)
        )
    if thickness_prior_stages is not None:
        archive_values["thickness_prior_stages"] = np.asarray(str(thickness_prior_stages))
    if warmup_prior_root is not None:
        archive_values["warmup_prior_root"] = np.asarray(
            portable_path_string(warmup_prior_root)
        )
    if constant_rate_warmup_diagnostics is not None:
        archive_values.update(
            {
                "constant_rate_map_rate_nm_day": np.asarray(
                    constant_rate_warmup_diagnostics[
                        "constant_rate_map_rate_nm_day"
                    ]
                ),
                "constant_rate_map_objective": np.asarray(
                    constant_rate_warmup_diagnostics["residual_cost"]
                ),
                "constant_rate_map_initial_objective": np.asarray(
                    constant_rate_warmup_diagnostics["initial_residual_cost"]
                ),
                "constant_rate_map_nfev": np.asarray(
                    constant_rate_warmup_diagnostics["nfev"]
                ),
            }
        )
    if fit_led_log_offsets:
        archive_values["led_log_offsets"] = np.asarray(
            values["led_log_offsets"], dtype=float
        )
    if reset_days.size > 1:
        archive_values.update(
            {
                "ice_reset_days": np.asarray(reset_days, dtype=float),
            }
        )
        if data["reset_event_names"][1] == "third_decon":
            archive_values.update(
                {
                    "third_decon_day": np.asarray(float(reset_days[1])),
                    "third_decon_initial_thickness_log_nm": np.asarray(
                        values["third_decon_initial_thickness_log_nm"]
                    ),
                    "post_third_decon_tile_log_offset": np.asarray(
                        values["post_third_decon_tile_log_offset"]
                    ),
                }
            )
        if (
            third_decon_initial_thickness_start_nm is not None
            and data["reset_event_names"][1] == "third_decon"
        ):
            archive_values["third_decon_initial_thickness_start_nm"] = np.asarray(
                float(third_decon_initial_thickness_start_nm)
            )
    archive_values["reset_event_days"] = np.asarray(reset_days, dtype=np.float64)
    archive_values["reset_event_names"] = np.asarray(
        data["reset_event_names"], dtype="U64"
    )
    archive_values["reset_event_thickness_site_names"] = np.asarray(
        thickness_names[1:], dtype="U96"
    )
    archive_values["reset_event_offset_site_names"] = np.asarray(
        offset_names, dtype="U96"
    )
    archive_values["reset_event_initial_thickness_log_nm"] = np.stack(
        [np.asarray(values[name], dtype=float) for name in thickness_names[1:]], axis=0
    )
    archive_values["reset_event_tile_log_offsets"] = np.stack(
        [np.asarray(values[name], dtype=float) for name in offset_names], axis=0
    )
    reset_event_starts = reset_event_initial_thickness_starts_nm or {}
    archive_values["reset_event_initial_thickness_start_nm"] = np.asarray(
        [reset_event_starts.get(name, np.nan) for name in data["reset_event_names"]],
        dtype=np.float64,
    )
    configured_events = data.get("reset_events", ())
    archive_values["reset_event_thickness_prior_median_nm"] = np.asarray(
        [
            event.get("thickness_prior", {}).get(
                "median_nm", production.INITIAL_THICKNESS_LOG_MEDIAN_NM
            )
            for event in configured_events
        ],
        dtype=np.float64,
    )
    archive_values["reset_event_thickness_prior_log_sigma"] = np.asarray(
        [
            event.get("thickness_prior", {}).get(
                "log_sigma", production.INITIAL_THICKNESS_LOG_SIGMA
            )
            for event in configured_events
        ],
        dtype=np.float64,
    )
    np.savez_compressed(output_dir / "posterior_samples.npz", **archive_values)

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
    active_reset_days = reset_days[
        (reset_days > np.min(days)) & (reset_days < np.max(days))
    ]
    boundaries = np.concatenate(([np.min(days)], active_reset_days, [np.max(days)]))
    accumulations = []
    for index in range(boundaries.size - 1):
        left, right = boundaries[index : index + 2]
        mask = (
            (node_days >= left) & (node_days <= right)
            if index == boundaries.size - 2
            else (node_days >= left) & (node_days < right)
        )
        indices = np.flatnonzero(mask)
        if indices.size < 2:
            raise ValueError("Relative-response trajectory lacks nodes in a reset segment")
        accumulations.append(node_ice[indices[-1]] - node_ice[indices[0]])
    mean_rate = float(np.sum(accumulations) / np.sum(np.diff(boundaries)))
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
            f"- Fit mode: `{fit_mode}`.",
            f"- Response cache: `{cache_path.resolve()}`.",
            f"- Evaluations: `{diagnostics['nfev']}`.",
            f"- Converged: `{diagnostics['converged']}`.",
            f"- Runtime: `{diagnostics['total_seconds']:.6g}` seconds.",
            f"- Free parameters: `{diagnostics['parameter_count']}`.",
            f"- Residual RMS excluding LED reference groups: `{np.sqrt(np.mean(residual_values**2)):.12g}` log.",
            f"- Maximum absolute residual excluding LED reference groups: `{np.max(np.abs(residual_values)):.12g}` log.",
            f"- Warm-start rate: `{diagnostics['trajectory_rate_initialization_nm_day']:.12g}` nm/day.",
            *(
                (
                    "- Constant-rate warm-up: `True`.",
                    f"- Constant-rate warm-up rate: `{constant_rate_warmup_diagnostics['constant_rate_map_rate_nm_day']:.12g}` nm/day.",
                    f"- Constant-rate warm-up objective: `{constant_rate_warmup_diagnostics['residual_cost']:.12g}`.",
                )
                if constant_rate_warmup_diagnostics is not None
                else ("- Constant-rate warm-up: `False`.",)
            ),
            f"- Final rate: `{rates[-1]:.12g}` nm/day.",
            f"- Mean rate: `{mean_rate:.12g}` nm/day.",
            f"- Physical reset days: `{reset_days.tolist()}`.",
            *(
                (
                    f"- Third-decon initial thickness start: "
                    f"`{float(third_decon_initial_thickness_start_nm):.12g}` nm.",
                )
                if third_decon_initial_thickness_start_nm is not None
                else ()
            ),
            f"- Trajectory prior: `{PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY}` nm/day, log sigma `{production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG}`.",
            f"- Trajectory entropy weight: `{PRODUCTION_ENTROPY_WEIGHT}`.",
            f"- LED log offsets fitted: `{bool(fit_led_log_offsets)}`.",
            *(
                (
                    f"- Warmup-centered thickness prior log sigma: `{float(thickness_prior_log_sigma):.12g}`.",
                    f"- Warmup-centered thickness prior stages: `{thickness_prior_stages}`.",
                )
                if thickness_prior_log_sigma is not None
                else ()
            ),
            *(
                (
                    f"- Warmup-centered thickness prior root: "
                    f"`{Path(warmup_prior_root).resolve()}`.",
                )
                if warmup_prior_root is not None
                else ()
            ),
            *(
                (
                    f"- LED log-offset prior log sigma: `{float(led_log_offset_prior_sigma_log):.12g}`.",
                    f"- LED log offsets: `{np.asarray(values['led_log_offsets'], dtype=float).reshape(-1).tolist()}`.",
                )
                if fit_led_log_offsets
                else ()
            ),
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
    third_decon_day=None,
    fit_led_log_offsets=False,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    third_decon_initial_thickness_start_nm=None,
    warmup_prior_root=None,
    thickness_prior_log_sigma=None,
    thickness_prior_stages=None,
    reset_events=None,
    reset_event_initial_thickness_starts_nm=None,
):
    """Fit one tile from a cached response and a previous tile fit."""
    (
        warmup_prior_root,
        thickness_prior_log_sigma,
        thickness_prior_stages,
    ) = resolve_warmup_prior_settings(
        third_decon_day,
        warmup_prior_root,
        thickness_prior_log_sigma,
        thickness_prior_stages,
    )
    jax, jnp, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    statistics_path = Path(statistics_path)
    response_cache_path = Path(response_cache_path)
    data = production.load_tile_statistics(
        statistics_path,
        third_decon_day=(None if reset_events is not None else third_decon_day),
        reset_events=reset_events,
    )
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
        fit_led_log_offsets=fit_led_log_offsets,
        third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
        reset_event_initial_thickness_starts_nm=(
            reset_event_initial_thickness_starts_nm
        ),
    )
    warmup_evaluator = build_ice_only_constant_rate_evaluator(
        data,
        values,
        cache,
        jax,
        jnp,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    values, warmup_diagnostics = fit_ice_only_constant_rate_reusable(
        data,
        cache,
        values,
        warmup_evaluator,
        jax,
        jnp,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    thickness_names, _ = production.reset_site_names(data)
    warmup_thickness_centers = {
        name: np.asarray(np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM))
        for name in thickness_names
    }
    warmup_thickness_sigmas = {
        name: production.INITIAL_THICKNESS_LOG_SIGMA for name in thickness_names
    }
    for event_index, name in enumerate(thickness_names[1:]):
        configured_events = data.get("reset_events", ())
        prior = (
            configured_events[event_index].get("thickness_prior", {})
            if len(configured_events) > event_index
            else {}
        )
        warmup_thickness_centers[name] = np.asarray(
            np.log(
                float(
                    prior.get(
                        "median_nm", production.INITIAL_THICKNESS_LOG_MEDIAN_NM
                    )
                )
            )
        )
        warmup_thickness_sigmas[name] = float(
            prior.get("log_sigma", production.INITIAL_THICKNESS_LOG_SIGMA)
        )
    if warmup_prior_root is not None:
        if thickness_prior_log_sigma is None or thickness_prior_stages is None:
            raise ValueError(
                "warmup_prior_root requires thickness_prior_log_sigma and "
                "thickness_prior_stages"
            )
        configured_centers, configured_sigmas = warmup_thickness_prior(
            warmup_prior_root,
            int(np.asarray(data["tile_ids"]).reshape(-1)[0]),
            data,
            float(thickness_prior_log_sigma),
            thickness_prior_stages,
        )
        warmup_thickness_centers.update(configured_centers)
        warmup_thickness_sigmas.update(configured_sigmas)
    elif thickness_prior_log_sigma is not None or thickness_prior_stages is not None:
        raise ValueError(
            "thickness prior settings require warmup_prior_root"
        )
    values, diagnostics = fit_ice_only(
        data,
        cache,
        values,
        jax,
        jnp,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        thickness_prior_log_centers=warmup_thickness_centers,
        thickness_prior_log_sigma=warmup_thickness_sigmas,
    )
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
        fit_led_log_offsets=fit_led_log_offsets,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        third_decon_initial_thickness_start_nm=third_decon_initial_thickness_start_nm,
        constant_rate_warmup_diagnostics=warmup_diagnostics,
        fit_mode=(
            "relative_response_warmup_centered_thickness_prior"
            if warmup_prior_root is not None
            else "relative_response_production"
        ),
        thickness_prior_log_sigma=thickness_prior_log_sigma,
        thickness_prior_stages=thickness_prior_stages,
        warmup_prior_root=warmup_prior_root,
        reset_event_initial_thickness_starts_nm=(
            reset_event_initial_thickness_starts_nm
        ),
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
        third_decon_day=args.third_decon_day,
        fit_led_log_offsets=args.fit_led_log_offsets,
        led_log_offset_prior_sigma_log=args.led_log_offset_prior_sigma_log,
        third_decon_initial_thickness_start_nm=args.third_decon_initial_thickness_start_nm,
        warmup_prior_root=args.warmup_prior_root,
        thickness_prior_log_sigma=args.thickness_prior_log_sigma,
        thickness_prior_stages=args.thickness_prior_stages,
    )


if __name__ == "__main__":
    main()
