"""Reusable per-tile relative optical-response caches for ice-only fits."""

from hashlib import sha256
import csv
from pathlib import Path
import time

import numpy as np

import fit_wfi_tmm_production as production


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 2
SPECTRAL_CACHE_VERSION = 1
MODEL_PROVENANCE_FIELDS = (
    "absolute_substrate_power",
    "absolute_hgcdte_tile_variation",
    "sed_mode",
    "sed_weighting",
    "wavelength_stride",
    "fully_np_trajectory",
    "whiten_np_trajectory",
    "fit_post_reset_tile_log_offset",
    "fit_incidence_angle",
    "incidence_pupil_half_angle_deg",
    "fit_ice_n_constant",
    "ice_n_constant_reference",
    "ice_n_constant_prior_sigma_log",
    "fit_ar_hgcdte_modulation",
    "ar_hgcdte_modulation_prior_sigma_log",
    "input_corrections_already_applied",
    "map_optimizer",
    "map_steps",
    "map_parameter_scaling",
    "trajectory_regularization",
    "trajectory_smoothness_weight",
    "trajectory_entropy_weight",
    "constant_rate_map_warmup",
    "tile_initial_thickness_log_median_nm",
    "tile_initial_thickness_log_sigma",
    "trajectory_rate_prior_mean_nm_day",
    "trajectory_rate_prior_sigma_log",
    "ice_reset_day",
)
OPTICAL_SITES = (
    "ar_tile_raw",
    "absolute_hgcdte_n_log_scale",
    "absolute_hgcdte_k_log_scale",
    "ice_n_log_constant",
    "ar_n_log_modulation",
    "hgcdte_n_log_modulation",
    "hgcdte_k_log_modulation",
    "incidence_angle_deg",
)
VARIABLE_SITES = (
    "initial_thickness_log_nm",
    "reset_initial_thickness_log_nm",
    "post_reset_tile_log_offset",
    "np_log_rate_nodes",
)


def resolve_portable_path(value, repository_root=None):
    """Resolve a cached path after its repository has moved to another machine.

    Production cache metadata historically stored absolute paths. If that path
    is no longer present, retain its dated run-relative suffix and resolve it
    beneath the current repository root.
    """
    path = Path(str(value)).expanduser()
    root = Path(repository_root or REPOSITORY_ROOT).resolve()
    candidates = []
    relocated_candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((root / path, path))
    for index, part in enumerate(path.parts):
        if len(part) == 8 and part.isdigit():
            relocated = root.joinpath(*path.parts[index:])
            candidates.append(relocated)
            relocated_candidates.append(relocated)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if relocated_candidates:
        return relocated_candidates[0].resolve()
    return (path if path.is_absolute() else root / path).resolve()


def portable_path_string(value, repository_root=None):
    """Serialize repository inputs relatively when they are inside the checkout."""
    root = Path(repository_root or REPOSITORY_ROOT).resolve()
    resolved = resolve_portable_path(value, repository_root=root)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def variable_sites(values):
    """Return the ice-only sites present in a reset-configured fit."""
    reset_sites = tuple(
        name
        for name in values
        if name.startswith(("reset_", "third_decon_", "post_reset_", "post_third_decon_"))
        and name not in VARIABLE_SITES
    )
    names = VARIABLE_SITES + reset_sites
    if "led_log_offsets" in values:
        names += ("led_log_offsets",)
    return names
# These are the production settings used by run_wfi_production.py.  The
# cached branch fixes the optical state but must retain the current-data ice
# trajectory prior rather than inheriting the source fit's trajectory.
PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY = 12.0
PRODUCTION_ENTROPY_WEIGHT = 0.0


def context_fingerprint(context):
    digest = sha256()
    for name in ("wavelengths", "model_weights", "qe_columns"):
        if name not in context:
            continue
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(context[name]).tobytes())
    return digest.hexdigest()


def file_fingerprint(path):
    """Return a content fingerprint for a provenance-controlled input file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def statistics_fingerprint(path):
    """Fingerprint the sufficient statistics used by a cached fit."""
    with Path(path).open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["group"]))
    if not rows:
        raise ValueError(f"No statistics rows found in {path}")
    digest = sha256()
    for field in ("n_pair_valid", "sum_log_change", "sum_squared_log_change"):
        if field not in rows[0]:
            raise ValueError(f"Missing {field} in {path}")
        digest.update(
            np.asarray([float(row[field]) for row in rows], dtype=np.float64).tobytes()
        )
    return digest.hexdigest()


def model_provenance_fingerprint(fit_archive):
    """Fingerprint the production settings embedded in a source fit."""
    digest = sha256()
    with np.load(Path(fit_archive), allow_pickle=False) as source:
        for name in MODEL_PROVENANCE_FIELDS:
            if name not in source:
                raise ValueError(f"{name} is missing from {fit_archive}")
            value = np.asarray(source[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def source_provenance(source_fit, source_statistics):
    """Collect and validate provenance from one production fit and its CSV."""
    source_fit = resolve_portable_path(source_fit)
    source_statistics = resolve_portable_path(source_statistics)
    with np.load(source_fit, allow_pickle=False) as source:
        if "statistics_path" not in source or "statistics_fingerprint" not in source:
            raise ValueError(f"Production provenance is missing from {source_fit}")
        archived_statistics = resolve_portable_path(
            str(np.asarray(source["statistics_path"]).reshape(-1)[0])
        )
        if archived_statistics != source_statistics:
            raise ValueError(
                f"Source fit statistics do not match cache statistics: "
                f"{archived_statistics} != {source_statistics}"
            )
        archived_statistics_fingerprint = str(
            np.asarray(source["statistics_fingerprint"]).reshape(-1)[0]
        )
    calculated_statistics_fingerprint = statistics_fingerprint(source_statistics)
    if calculated_statistics_fingerprint != archived_statistics_fingerprint:
        raise ValueError(
            f"Source fit statistics fingerprint does not match {source_statistics}"
        )
    return {
        "source_fit": portable_path_string(source_fit),
        "source_statistics": portable_path_string(source_statistics),
        "source_fit_sha256": file_fingerprint(source_fit),
        "source_statistics_sha256": file_fingerprint(source_statistics),
        "source_statistics_fingerprint": archived_statistics_fingerprint,
        "source_model_fingerprint": model_provenance_fingerprint(source_fit),
    }


def _strip_leading_axes(value, target_shape):
    value = np.asarray(value, dtype=float)
    while value.ndim > len(target_shape) and value.shape[0] == 1:
        value = value[0]
    if value.shape != tuple(target_shape):
        raise ValueError(f"Reference shape {value.shape} does not match {target_shape}")
    return value.copy()


def load_warm_values(
    data,
    reference_fit,
    reference_statistics=None,
    trajectory_rate_initialization_nm_day=PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY,
    fit_led_log_offsets=False,
    third_decon_initial_thickness_start_nm=None,
    reset_event_initial_thickness_starts_nm=None,
):
    """Load source optical values and current-data ice initial values.

    The source fit supplies only the fixed optical state used to build the
    response cache.  Its trajectory is deliberately not reused as a prior.
    """
    if fit_led_log_offsets:
        values = production.initial_values_for_data(
            data,
            constant_rate=False,
            fit_led_log_offsets=True,
        )
    else:
        values = production.initial_values_for_data(data, constant_rate=False)
    if (
        not np.isfinite(trajectory_rate_initialization_nm_day)
        or trajectory_rate_initialization_nm_day <= 0.0
    ):
        raise ValueError("Trajectory-rate initialization must be positive and finite")
    thickness_names, offset_names = production.reset_site_names(data)
    with np.load(Path(reference_fit), allow_pickle=False) as source:
        source_names = (
            "ar_tile_raw",
            "initial_thickness_log_nm",
            "absolute_hgcdte_n_log_scale",
            "absolute_hgcdte_k_log_scale",
            *thickness_names[1:],
            *offset_names,
        )
        for name in source_names:
            if name in source:
                values[name] = _strip_leading_axes(source[name], np.asarray(values[name]).shape)
        if "incidence_angle_deg" in values:
            values["incidence_angle_deg"] = (
                float(np.asarray(source["incidence_angle_deg"]).reshape(-1)[0])
                if "incidence_angle_deg" in source
                else 0.0
            )
    # Keep the source-statistics argument for cache provenance validation, but
    # never use its old trajectory to constrain the current data.
    del reference_statistics
    starts = dict(reset_event_initial_thickness_starts_nm or {})
    if third_decon_initial_thickness_start_nm is not None:
        third_decon_initial_thickness_start_nm = float(
            third_decon_initial_thickness_start_nm
        )
        if (
            not np.isfinite(third_decon_initial_thickness_start_nm)
            or third_decon_initial_thickness_start_nm <= 0.0
        ):
            raise ValueError(
                "Third-decon initial thickness start must be positive and finite"
            )
        if "third_decon_initial_thickness_log_nm" not in values:
            raise ValueError(
                "Third-decon initial thickness start requires a third-decon fit"
            )
        starts["third_decon"] = third_decon_initial_thickness_start_nm
    thickness_names, _ = production.reset_site_names(data)
    event_names = production._reset_event_names(data, len(thickness_names) - 1)
    for index, (event_name, site_name) in enumerate(
        zip(event_names, thickness_names[1:], strict=True)
    ):
        if event_name not in starts:
            continue
        start_nm = float(starts[event_name])
        if not np.isfinite(start_nm) or start_nm <= 0.0:
            raise ValueError(f"Reset event {event_name} thickness start must be positive")
        values[site_name] = np.full_like(
            np.asarray(values[site_name], dtype=float), np.log(start_nm)
        )
    if (
        third_decon_initial_thickness_start_nm is not None
        and "third_decon_initial_thickness_log_nm" in values
        and "third_decon" not in event_names
    ):
        values["third_decon_initial_thickness_log_nm"] = np.full_like(
            np.asarray(values["third_decon_initial_thickness_log_nm"], dtype=float),
            np.log(third_decon_initial_thickness_start_nm),
        )
    values["np_log_rate_nodes"] = np.full_like(
        np.asarray(values["np_log_rate_nodes"], dtype=float),
        np.log(trajectory_rate_initialization_nm_day),
    )
    return values


def rate_prior_reference(initial_values):
    """Return the fixed production rate-prior center, independent of warm starts."""
    return np.full_like(
        np.asarray(initial_values["np_log_rate_nodes"], dtype=float),
        np.log(PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY),
    )


def _led_log_offset_values(data, values, jnp):
    """Return fitted post-reference LED offsets for each observation group."""
    if "led_log_offsets" not in values:
        return 0.0
    offsets = jnp.asarray(values["led_log_offsets"])
    offsets = offsets.reshape((-1, offsets.shape[-1]))
    led_index = jnp.asarray(np.asarray(data["led"], dtype=int) - 1, dtype=jnp.int32)
    nonreference = jnp.asarray(np.asarray(data["repeat"], dtype=int) != 0)
    return offsets[:, led_index] * nonreference[None, :]


def load_optical_values(fit_archive):
    fit_archive = Path(fit_archive)
    with np.load(fit_archive, allow_pickle=False) as source:
        values = {
            name: np.asarray(source[name], dtype=float).copy()
            for name in OPTICAL_SITES
            if name in source
        }
        missing = [name for name in OPTICAL_SITES if name not in values]
        response_cache = source.get("response_cache") if missing else None

    if missing and response_cache is not None:
        response_cache_path = Path(str(np.asarray(response_cache).reshape(-1)[0]))
        if not response_cache_path.is_absolute():
            response_cache_path = fit_archive.parent / response_cache_path
        if response_cache_path.is_file():
            with np.load(response_cache_path, allow_pickle=False) as cache:
                for name in missing:
                    if name in cache:
                        values[name] = np.asarray(cache[name], dtype=float).copy()

    for name in OPTICAL_SITES:
        if name not in values:
            if name == "incidence_angle_deg":
                # Current production fixes incidence at normal and omits this site.
                values[name] = np.asarray(0.0)
                continue
            raise ValueError(f"{name} is missing from {fit_archive}")
    return values


def _relative_lookup_data(data):
    selected = []
    for led in range(1, 7):
        indices = np.flatnonzero(np.asarray(data["led"]) == led)
        if indices.size < 2:
            raise ValueError(f"LED {led} has no non-reference group")
        selected.extend((int(indices[0]), int(indices[1])))
    selected = np.asarray(selected, dtype=int)
    mini = dict(data)
    mini["counts"] = np.ones((1, selected.size), dtype=float)
    mini["sums"] = np.zeros((1, selected.size), dtype=float)
    mini["sum_squared"] = np.zeros((1, selected.size), dtype=float)
    for name in ("groups", "led", "repeat", "program", "days", "exposure_times"):
        mini[name] = np.asarray(data[name])[selected]
    mini["first_by_led"] = np.arange(0, 12, 2, dtype=int)
    mini["repeat_values"] = np.asarray(sorted(set(mini["repeat"].tolist())), dtype=int)
    mini["repeat_index"] = np.zeros(12, dtype=int)
    return mini


def build_response_lookup(
    data,
    context,
    optical_values,
    grid_nm,
    compiled_predictor=None,
):
    """Evaluate six fixed-optics response curves relative to zero thickness."""
    jax, jnp = production.require_jax()
    grid_nm = np.asarray(grid_nm, dtype=float).reshape(-1)
    if grid_nm.size < 2 or np.any(~np.isfinite(grid_nm)) or np.any(np.diff(grid_nm) <= 0.0):
        raise ValueError("Response grid must contain increasing finite values")

    mini = _relative_lookup_data(data)
    if compiled_predictor is None:
        predictor = production.build_prediction_function(
            mini,
            context,
            fit_ice_n_constant=True,
            fit_ar_hgcdte_modulation=True,
        )
        compiled_predictor = jax.jit(predictor)
    ar_raw = np.asarray(optical_values["ar_tile_raw"], dtype=float).reshape(-1)
    ar_physical = np.asarray(
        production._physical_ar_parameters(jnp.asarray(ar_raw[None, :]), jax, jnp)
    )
    hgcdte = np.asarray(
        [
            float(np.asarray(optical_values["absolute_hgcdte_n_log_scale"]).reshape(-1)[0]),
            float(np.asarray(optical_values["absolute_hgcdte_k_log_scale"]).reshape(-1)[0]),
        ]
    )
    ice_n_log_constant = np.asarray(
        optical_values["ice_n_log_constant"], dtype=float
    ).reshape(-1)
    ar_n_log_modulation = np.asarray(
        optical_values["ar_n_log_modulation"], dtype=float
    ).reshape(-1, production.AR_LAYERS, len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM))
    hgcdte_n_log_modulation = np.asarray(
        optical_values["hgcdte_n_log_modulation"], dtype=float
    ).reshape(-1, len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM))
    hgcdte_k_log_modulation = np.asarray(
        optical_values["hgcdte_k_log_modulation"], dtype=float
    ).reshape(-1, len(production.ICE_N_MODULATION_NODE_WAVELENGTHS_NM))
    row_count = 6 * grid_nm.size
    ice = np.zeros((row_count, 12), dtype=float)
    for led_index in range(6):
        start = led_index * grid_nm.size
        stop = (led_index + 1) * grid_nm.size
        ice[start:stop, 2 * led_index + 1] = grid_nm
    ar_rows = np.broadcast_to(ar_physical, (row_count, ar_physical.shape[-1]))
    hg_rows = np.broadcast_to(hgcdte, (row_count, 2))
    ice_n_rows = np.broadcast_to(ice_n_log_constant, (row_count,))
    ar_n_rows = np.broadcast_to(ar_n_log_modulation, (row_count,) + ar_n_log_modulation.shape[1:])
    hgcdte_n_rows = np.broadcast_to(
        hgcdte_n_log_modulation,
        (row_count,) + hgcdte_n_log_modulation.shape[1:],
    )
    hgcdte_k_rows = np.broadcast_to(
        hgcdte_k_log_modulation,
        (row_count,) + hgcdte_k_log_modulation.shape[1:],
    )
    started = time.perf_counter()
    prediction = np.asarray(
        compiled_predictor(
            jnp.asarray(ice),
            jnp.asarray(ar_rows),
            jnp.asarray(hg_rows),
            float(np.asarray(optical_values["incidence_angle_deg"]).reshape(-1)[0]),
            ice_n_log_constant=jnp.asarray(ice_n_rows),
            ar_n_log_modulation=jnp.asarray(ar_n_rows),
            hgcdte_n_log_modulation=jnp.asarray(hgcdte_n_rows),
            hgcdte_k_log_modulation=jnp.asarray(hgcdte_k_rows),
        )
    )
    curves = np.empty((6, grid_nm.size), dtype=float)
    for led_index in range(6):
        start = led_index * grid_nm.size
        stop = (led_index + 1) * grid_nm.size
        curves[led_index] = prediction[start:stop, 2 * led_index + 1]
    return curves, time.perf_counter() - started, compiled_predictor


def write_response_cache(
    path,
    data,
    context,
    optical_values,
    grid_nm,
    curves,
    source_fit,
    source_statistics,
    build_seconds,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tile_id = int(np.asarray(data["tile_ids"]).reshape(-1)[0])
    detector = str(np.asarray(data["sca_names"]).reshape(-1)[0])
    provenance = source_provenance(source_fit, source_statistics)
    np.savez_compressed(
        path,
        cache_version=np.asarray(CACHE_VERSION),
        detector=np.asarray(detector),
        tile_id=np.asarray(tile_id),
        grid_nm=np.asarray(grid_nm, dtype=float),
        relative_log_response=np.asarray(curves, dtype=float),
        ar_tile_raw=np.asarray(optical_values["ar_tile_raw"], dtype=float),
        absolute_hgcdte_n_log_scale=np.asarray(
            optical_values["absolute_hgcdte_n_log_scale"], dtype=float
        ),
         absolute_hgcdte_k_log_scale=np.asarray(
            optical_values["absolute_hgcdte_k_log_scale"], dtype=float
        ),
        ice_n_log_constant=np.asarray(optical_values["ice_n_log_constant"], dtype=float),
        ar_n_log_modulation=np.asarray(
            optical_values["ar_n_log_modulation"], dtype=float
        ),
        hgcdte_n_log_modulation=np.asarray(
            optical_values["hgcdte_n_log_modulation"], dtype=float
        ),
        hgcdte_k_log_modulation=np.asarray(
            optical_values["hgcdte_k_log_modulation"], dtype=float
        ),
        incidence_angle_deg=np.asarray(optical_values["incidence_angle_deg"], dtype=float),
        context_fingerprint=np.asarray(context_fingerprint(context)),
        source_fit=np.asarray(provenance["source_fit"]),
        source_statistics=np.asarray(provenance["source_statistics"]),
        source_fit_sha256=np.asarray(provenance["source_fit_sha256"]),
        source_statistics_sha256=np.asarray(provenance["source_statistics_sha256"]),
        source_statistics_fingerprint=np.asarray(
            provenance["source_statistics_fingerprint"]
        ),
        source_model_fingerprint=np.asarray(provenance["source_model_fingerprint"]),
        build_seconds=np.asarray(float(build_seconds)),
    )


def load_response_cache(path, data, context):
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        version = int(np.asarray(source["cache_version"]).reshape(-1)[0])
        if version != CACHE_VERSION:
            raise ValueError(f"Unsupported response cache version {version}")
        detector = str(np.asarray(source["detector"]).reshape(-1)[0])
        tile_id = int(np.asarray(source["tile_id"]).reshape(-1)[0])
        expected_detector = str(np.asarray(data["sca_names"]).reshape(-1)[0])
        expected_tile = int(np.asarray(data["tile_ids"]).reshape(-1)[0])
        if detector != expected_detector or tile_id != expected_tile:
            raise ValueError(
                f"Response cache identifies {detector} tile {tile_id}, "
                f"not {expected_detector} tile {expected_tile}"
            )
        expected_context = context_fingerprint(context)
        cached_context = str(np.asarray(source["context_fingerprint"]).reshape(-1)[0])
        if cached_context != expected_context:
            raise ValueError("Response cache does not match the current optical context")
        grid = np.asarray(source["grid_nm"], dtype=float)
        curves = np.asarray(source["relative_log_response"], dtype=float)
        if grid.ndim != 1 or curves.shape != (6, grid.size):
            raise ValueError("Response cache has invalid six-LED response shape")
        return {
            "grid_nm": grid,
            "relative_log_response": curves,
            "source_fit": str(
                resolve_portable_path(
                    str(np.asarray(source["source_fit"]).reshape(-1)[0])
                )
            ),
            "source_statistics": str(
                resolve_portable_path(
                    str(np.asarray(source["source_statistics"]).reshape(-1)[0])
                )
            ),
            "source_fit_sha256": str(
                np.asarray(source["source_fit_sha256"]).reshape(-1)[0]
            ),
            "source_statistics_sha256": str(
                np.asarray(source["source_statistics_sha256"]).reshape(-1)[0]
            ),
            "source_statistics_fingerprint": str(
                np.asarray(source["source_statistics_fingerprint"]).reshape(-1)[0]
            ),
            "source_model_fingerprint": str(
                np.asarray(source["source_model_fingerprint"]).reshape(-1)[0]
            ),
            "build_seconds": float(np.asarray(source["build_seconds"]).reshape(-1)[0]),
        }


def validate_response_cache_provenance(
    path,
    expected_fit=None,
    expected_statistics=None,
):
    """Reject caches whose source files or production metadata have changed."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        fields = (
            "source_fit",
            "source_statistics",
            "source_fit_sha256",
            "source_statistics_sha256",
            "source_statistics_fingerprint",
            "source_model_fingerprint",
        )
        missing = [name for name in fields if name not in source]
        if missing:
            raise ValueError(
                f"Response cache lacks provenance fields: {', '.join(missing)}"
            )
        source_fit = resolve_portable_path(
            str(np.asarray(source["source_fit"]).reshape(-1)[0])
        )
        source_statistics = resolve_portable_path(
            str(np.asarray(source["source_statistics"]).reshape(-1)[0])
        )
        if expected_fit is not None and source_fit != Path(expected_fit).resolve():
            raise ValueError(
                f"Response cache source fit is {source_fit}, expected {Path(expected_fit).resolve()}"
            )
        if expected_statistics is not None and source_statistics != Path(expected_statistics).resolve():
            raise ValueError(
                "Response cache source statistics are "
                f"{source_statistics}, expected {Path(expected_statistics).resolve()}"
            )
        cached_fit_sha = str(np.asarray(source["source_fit_sha256"]).reshape(-1)[0])
        cached_statistics_sha = str(
            np.asarray(source["source_statistics_sha256"]).reshape(-1)[0]
        )
        cached_statistics_fingerprint = str(
            np.asarray(source["source_statistics_fingerprint"]).reshape(-1)[0]
        )
        cached_model_fingerprint = str(
            np.asarray(source["source_model_fingerprint"]).reshape(-1)[0]
        )
    if not source_fit.is_file() or not source_statistics.is_file():
        raise FileNotFoundError(
            f"Response cache provenance files are missing for {path}"
        )
    current = source_provenance(source_fit, source_statistics)
    if current["source_fit_sha256"] != cached_fit_sha:
        raise ValueError(f"Response cache source fit has changed: {source_fit}")
    if current["source_statistics_sha256"] != cached_statistics_sha:
        raise ValueError(
            f"Response cache source statistics have changed: {source_statistics}"
        )
    if current["source_statistics_fingerprint"] != cached_statistics_fingerprint:
        raise ValueError("Response cache statistics fingerprint does not match its source fit")
    if current["source_model_fingerprint"] != cached_model_fingerprint:
        raise ValueError("Response cache production-model provenance does not match its source fit")
    return current


def relative_state(data, values, cache, jax, jnp):
    """Evaluate ice trajectory and cached relative response."""
    thickness_names, _ = production.reset_site_names(data)
    reset_initial = (
        jnp.stack([values[name] for name in thickness_names[1:]], axis=0)
        if len(thickness_names) > 2
        else values[thickness_names[1]]
    )
    ice, rates, node_ice, node_rates = production._trajectory_state(
        data,
        values["np_log_rate_nodes"],
        values["initial_thickness_log_nm"],
        reset_initial,
        jax,
        jnp,
    )
    grid = jnp.asarray(cache["grid_nm"])
    curves = jnp.asarray(cache["relative_log_response"])
    group_index = jnp.arange(len(data["groups"]), dtype=jnp.int32)
    led_index = jnp.asarray(np.asarray(data["led"], dtype=int) - 1, dtype=jnp.int32)
    current_by_led = jnp.stack(
        [jnp.interp(ice[0], grid, curves[led]) for led in range(6)]
    )
    current = current_by_led[led_index, group_index]
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    baseline_thickness = ice[0, first_by_led][led_index]
    baseline_by_led = jnp.stack(
        [jnp.interp(baseline_thickness, grid, curves[led]) for led in range(6)]
    )
    baseline = baseline_by_led[led_index, group_index]
    prediction = (current - baseline)[None, :] + _led_log_offset_values(
        data,
        values,
        jnp,
    )
    return {
        "ice": ice,
        "rates": rates,
        "node_ice": node_ice,
        "node_rates": node_rates,
        "prediction": prediction,
    }


def _stack_tile_data(data_list):
    """Combine tiles that share the same observation layout for batch fitting."""
    if not data_list:
        raise ValueError("At least one tile is required for a batch fit")
    shared_names = (
        "days",
        "node_days",
        "led",
        "repeat",
        "program",
        "groups",
        "first_by_led",
        "repeat_values",
        "repeat_index",
        "exposure_times",
        "reset_day",
        "reset_days",
    )
    reference = data_list[0]
    for data in data_list[1:]:
        for name in shared_names:
            if not np.array_equal(np.asarray(reference[name]), np.asarray(data[name])):
                raise ValueError(
                    f"Cannot batch tiles with different {name} observation layouts"
                )
    batch = dict(reference)
    for name in ("counts", "sums", "sum_squared"):
        batch[name] = np.concatenate(
            [np.asarray(data[name], dtype=float) for data in data_list], axis=0
        )
    batch["tile_ids"] = np.concatenate(
        [np.asarray(data["tile_ids"], dtype=int) for data in data_list]
    )
    batch["n_tiles"] = 1
    batch["n_scas"] = 1
    return batch


def relative_state_batch(data, values, cache, jax, jnp):
    """Evaluate independent tile trajectories against a stacked cache."""
    thickness_names, _ = production.reset_site_names(data)
    reset_initial = (
        jnp.stack([values[name] for name in thickness_names[1:]], axis=0)
        if len(thickness_names) > 2
        else values[thickness_names[1]]
    )
    ice, rates, node_ice, node_rates = production._trajectory_state(
        data,
        values["np_log_rate_nodes"],
        values["initial_thickness_log_nm"],
        reset_initial,
        jax,
        jnp,
    )
    grid = jnp.asarray(cache["grid_nm"])
    curves = jnp.asarray(cache["relative_log_response"])

    def interpolate_by_led(thickness, tile_curves):
        return jnp.stack(
            [jnp.interp(thickness, grid, tile_curves[led]) for led in range(6)]
        )

    current_by_led = jax.vmap(interpolate_by_led)(ice, curves)
    group_index = jnp.arange(len(data["groups"]), dtype=jnp.int32)
    led_index = jnp.asarray(np.asarray(data["led"], dtype=int) - 1, dtype=jnp.int32)
    current = current_by_led[:, led_index, group_index]
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    baseline_thickness = ice[:, first_by_led][:, led_index]
    baseline_by_led = jax.vmap(interpolate_by_led)(baseline_thickness, curves)
    baseline = baseline_by_led[:, led_index, group_index]
    return {
        "ice": ice,
        "rates": rates,
        "node_ice": node_ice,
        "node_rates": node_rates,
        "prediction": current - baseline + _led_log_offset_values(data, values, jnp),
    }


def _observed_with_offset(data, values, jnp):
    observed = jnp.asarray(data["sums"]) / jnp.asarray(data["counts"])
    _, offset_names = production.reset_site_names(data)
    reset_mask = jnp.asarray(
        np.asarray(data["days"], dtype=float)[None, :]
        >= production.reset_days_for_data(data)[:, None]
    )
    reset_offsets = jnp.stack(
        [jnp.asarray(values[name]).reshape(-1) for name in offset_names], axis=0
    )
    return observed + jnp.sum(
        reset_offsets[:, :, None] * reset_mask[:, None, :], axis=0
    )


def _prior_residuals(
    data,
    values,
    rate_reference,
    jnp,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    thickness_prior_log_centers=None,
    thickness_prior_log_sigma=production.INITIAL_THICKNESS_LOG_SIGMA,
):
    thickness_names, offset_names = production.reset_site_names(data)

    def thickness_sigma(name):
        sigma = (
            thickness_prior_log_sigma[name]
            if isinstance(thickness_prior_log_sigma, dict)
            else thickness_prior_log_sigma
        )
        if isinstance(sigma, (int, float, np.integer, np.floating)) and (
            not np.isfinite(sigma) or sigma <= 0.0
        ):
            raise ValueError("Thickness prior log sigma must be positive and finite")
        return jnp.asarray(sigma, dtype=float)

    def thickness_center(name):
        if thickness_prior_log_centers is None:
            return np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
        if name not in thickness_prior_log_centers:
            raise ValueError(f"Missing thickness prior center for {name}")
        return jnp.asarray(thickness_prior_log_centers[name])

    residuals = [
        *[
            (
                jnp.asarray(values[name]) - thickness_center(name)
            ).reshape(-1)
            / thickness_sigma(name)
            for name in thickness_names
        ],
        (
            jnp.asarray(values["np_log_rate_nodes"])
            - jnp.asarray(rate_reference)
        ).reshape(-1)
        / production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        *(
            jnp.asarray(values[name]).reshape(-1)
            / production.POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA
            for name in offset_names
        ),
    ]
    if "led_log_offsets" in values:
        residuals.append(
            jnp.asarray(values["led_log_offsets"]).reshape(-1)
            / led_log_offset_prior_sigma_log
        )
    return jnp.concatenate(residuals)


def fit_ice_only(
    data,
    cache,
    initial_values,
    jax,
    jnp,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
    thickness_prior_log_centers=None,
    thickness_prior_log_sigma=production.INITIAL_THICKNESS_LOG_SIGMA,
):
    """Fit only initial/reset ice, offset, and non-parametric rate nodes."""
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    rate_reference = rate_prior_reference(initial_values)
    variable_site_names = variable_sites(initial_values)
    variable_initial = {
        name: np.asarray(initial_values[name], dtype=float).copy()
        for name in variable_site_names
    }
    fixed_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in initial_values.items()
        if name not in variable_site_names
    }
    flat_initial, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in variable_initial.items()}
    )
    parameter_scales = production.map_parameter_scale_values(
        initial_values,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    if thickness_prior_log_centers is not None:
        for name in production.reset_site_names(data)[0]:
            if name.endswith("_log_nm") and name in parameter_scales:
                sigma = (
                    thickness_prior_log_sigma[name]
                    if isinstance(thickness_prior_log_sigma, dict)
                    else thickness_prior_log_sigma
                )
                parameter_scales[name] = np.full_like(
                    np.asarray(parameter_scales[name], dtype=float),
                    sigma,
                )
    flat_scales, _ = ravel_pytree(
        {
            name: jnp.asarray(parameter_scales[name])
            for name in variable_site_names
        }
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)

    def full_values(scaled_values):
        values = dict(fixed_values)
        values.update(
            unravel(
                jnp.asarray(flat_initial) + jnp.asarray(flat_scales) * scaled_values
            )
        )
        return values

    def residuals(scaled_values):
        values = full_values(scaled_values)
        state = relative_state(data, values, cache, jax, jnp)
        observed = _observed_with_offset(data, values, jnp)
        data_residual = (
            jnp.sqrt(production.default_likelihood_weight(data["counts"]) / int(data["n_tiles"]))
            / production.LOG_NOISE_SIGMA
            * jnp.sqrt(jnp.asarray(data["counts"]))
            * (state["prediction"] - observed)
        ).reshape(-1)
        return jnp.concatenate(
            (
                data_residual,
                _prior_residuals(
                    data,
                    values,
                    rate_reference,
                    jnp,
                    led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
                    thickness_prior_log_centers=thickness_prior_log_centers,
                    thickness_prior_log_sigma=thickness_prior_log_sigma,
                ),
            )
        )

    residual_jit = jax.jit(residuals)
    jacobian_jit = jax.jit(jax.jacfwd(residuals))

    def scipy_residual(scaled_values):
        return np.asarray(residual_jit(jnp.asarray(scaled_values)), dtype=float)

    def scipy_jacobian(scaled_values):
        return np.asarray(jacobian_jit(jnp.asarray(scaled_values)), dtype=float)

    initial_scaled = np.zeros_like(flat_initial)
    started = time.perf_counter()
    initial_residual = scipy_residual(initial_scaled)
    compile_seconds = time.perf_counter() - started
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=production.PRODUCTION_MAP_STEPS,
        verbose=0,
    )
    total_seconds = time.perf_counter() - started
    values = full_values(np.asarray(result.x, dtype=float))
    final_residual = scipy_residual(np.asarray(result.x, dtype=float))
    return values, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameter_count": int(flat_initial.size),
        "compile_seconds": float(compile_seconds),
        "total_seconds": float(total_seconds),
        "residual_cost": float(0.5 * np.sum(final_residual**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_residual**2)),
        "trajectory_rate_initialization_nm_day": float(
            np.exp(np.median(variable_initial["np_log_rate_nodes"]))
        ),
    }


def fit_ice_only_constant_rate(
    data,
    cache,
    initial_values,
    jax,
    jnp,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    """Fit reset thicknesses and one scalar accumulation rate using a cache."""
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    initial_rates = np.asarray(initial_values["np_log_rate_nodes"], dtype=float)
    rate_shape = initial_rates.shape[:-1]
    variable_site_names = tuple(
        name for name in variable_sites(initial_values) if name != "np_log_rate_nodes"
    ) + ("constant_log_rate",)
    variable_initial = {
        name: np.asarray(initial_values[name], dtype=float).copy()
        for name in variable_site_names
        if name != "constant_log_rate"
    }
    variable_initial["constant_log_rate"] = np.full(
        rate_shape,
        float(np.median(initial_rates)),
        dtype=float,
    )
    fixed_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in initial_values.items()
        if name not in variable_sites(initial_values)
    }
    flat_initial, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in variable_initial.items()}
    )
    parameter_scales = production.map_parameter_scale_values(
        initial_values,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    scale_values = {
        name: np.asarray(parameter_scales[name], dtype=float)
        for name in variable_site_names
        if name != "constant_log_rate"
    }
    scale_values["constant_log_rate"] = np.full(
        rate_shape,
        production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        dtype=float,
    )
    flat_scales, _ = ravel_pytree(
        {name: jnp.asarray(value) for name, value in scale_values.items()}
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)

    def full_values(scaled_values):
        values = dict(fixed_values)
        values.update(
            unravel(
                jnp.asarray(flat_initial) + jnp.asarray(flat_scales) * scaled_values
            )
        )
        constant_log_rate = values.pop("constant_log_rate")
        values["np_log_rate_nodes"] = jnp.broadcast_to(
            jnp.asarray(constant_log_rate)[..., None],
            rate_shape + (initial_rates.shape[-1],),
        )
        return values

    def residuals(scaled_values):
        values = full_values(scaled_values)
        state = relative_state(data, values, cache, jax, jnp)
        observed = _observed_with_offset(data, values, jnp)
        data_residual = (
            jnp.sqrt(
                production.default_likelihood_weight(data["counts"])
                / int(data["n_tiles"])
            )
            / production.LOG_NOISE_SIGMA
            * jnp.sqrt(jnp.asarray(data["counts"]))
            * (state["prediction"] - observed)
        ).reshape(-1)
        thickness_names, offset_names = production.reset_site_names(data)
        prior_residuals = [
            (
                jnp.asarray(values["initial_thickness_log_nm"])
                - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
            ).reshape(-1)
            / production.INITIAL_THICKNESS_LOG_SIGMA,
            *[
                (
                    jnp.asarray(values[name])
                    - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
                ).reshape(-1)
                / production.INITIAL_THICKNESS_LOG_SIGMA
                for name in thickness_names[1:]
            ],
            (
                jnp.asarray(values["np_log_rate_nodes"])[..., 0]
                - np.log(PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY)
            ).reshape(-1)
            / production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
            *[
                jnp.asarray(values[name]).reshape(-1)
                / production.POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA
                for name in offset_names
            ],
        ]
        if "led_log_offsets" in values:
            prior_residuals.append(
                jnp.asarray(values["led_log_offsets"]).reshape(-1)
                / led_log_offset_prior_sigma_log
            )
        return jnp.concatenate((data_residual, *prior_residuals))

    residual_jit = jax.jit(residuals)
    jacobian_jit = jax.jit(jax.jacfwd(residuals))

    def scipy_residual(scaled_values):
        return np.asarray(residual_jit(jnp.asarray(scaled_values)), dtype=float)

    def scipy_jacobian(scaled_values):
        return np.asarray(jacobian_jit(jnp.asarray(scaled_values)), dtype=float)

    initial_scaled = np.zeros_like(flat_initial)
    started = time.perf_counter()
    initial_residual = scipy_residual(initial_scaled)
    compile_seconds = time.perf_counter() - started
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=production.PRODUCTION_MAP_STEPS,
        verbose=0,
    )
    total_seconds = time.perf_counter() - started
    values = full_values(np.asarray(result.x, dtype=float))
    final_residual = scipy_residual(np.asarray(result.x, dtype=float))
    rate_nm_day = float(np.exp(np.asarray(values["np_log_rate_nodes"]).reshape(-1)[0]))
    return values, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameter_count": int(flat_initial.size),
        "compile_seconds": float(compile_seconds),
        "total_seconds": float(total_seconds),
        "residual_cost": float(0.5 * np.sum(final_residual**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_residual**2)),
        "constant_rate_map_rate_nm_day": rate_nm_day,
        "trajectory_rate_initialization_nm_day": float(
            np.exp(np.median(initial_rates))
        ),
    }


def build_ice_only_constant_rate_evaluator(
    data,
    initial_values,
    cache,
    jax,
    jnp,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    """Compile a reusable evaluator for scalar-rate cached-response warmups."""
    from jax.flatten_util import ravel_pytree

    initial_rates = np.asarray(initial_values["np_log_rate_nodes"], dtype=float)
    rate_shape = initial_rates.shape[:-1]
    variable_site_names = tuple(
        name for name in variable_sites(initial_values) if name != "np_log_rate_nodes"
    ) + ("constant_log_rate",)
    variable_initial = {
        name: np.asarray(initial_values[name], dtype=float).copy()
        for name in variable_site_names
        if name != "constant_log_rate"
    }
    variable_initial["constant_log_rate"] = np.full(
        rate_shape,
        float(np.median(initial_rates)),
        dtype=float,
    )
    _, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in variable_initial.items()}
    )
    grid = jnp.asarray(np.asarray(cache["grid_nm"], dtype=float))
    days = jnp.asarray(data["days"])
    node_days = jnp.asarray(data["node_days"])
    led_index = jnp.asarray(np.asarray(data["led"], dtype=int) - 1, dtype=jnp.int32)
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    group_index = jnp.arange(len(data["groups"]), dtype=jnp.int32)
    thickness_names, offset_names = production.reset_site_names(data)
    reset_mask = jnp.asarray(
        np.asarray(data["days"], dtype=float)[None, :]
        >= production.reset_days_for_data(data)[:, None]
    )
    layout_data = {
        "days": np.asarray(data["days"], dtype=float),
        "node_days": np.asarray(data["node_days"], dtype=float),
        "reset_day": float(data["reset_day"]),
        "reset_days": production.reset_days_for_data(data),
    }

    def expand_values(values):
        values = dict(values)
        constant_log_rate = values.pop("constant_log_rate")
        values["np_log_rate_nodes"] = jnp.broadcast_to(
            jnp.asarray(constant_log_rate)[..., None],
            rate_shape + (initial_rates.shape[-1],),
        )
        return values

    def prediction(values, curves):
        values = expand_values(values)
        reset_initial = (
            jnp.stack([values[name] for name in thickness_names[1:]], axis=0)
            if len(thickness_names) > 2
            else values[thickness_names[1]]
        )
        ice, _, _, _ = production._trajectory_state(
            layout_data,
            values["np_log_rate_nodes"],
            values["initial_thickness_log_nm"],
            reset_initial,
            jax,
            jnp,
        )

        def interpolate_by_led(thickness):
            return jnp.stack(
                [jnp.interp(thickness, grid, curves[led]) for led in range(6)]
            )

        current_by_led = interpolate_by_led(ice[0])
        current = current_by_led[led_index, group_index]
        baseline_thickness = ice[0, first_by_led][led_index]
        baseline_by_led = interpolate_by_led(baseline_thickness)
        baseline = baseline_by_led[led_index, group_index]
        return current - baseline + _led_log_offset_values(data, values, jnp)[0]

    def residual_function(
        scaled_values,
        flat_initial,
        flat_scales,
        counts,
        sums,
        curves,
    ):
        values = unravel(flat_initial + flat_scales * scaled_values)
        expanded = expand_values(values)
        reset_offsets = jnp.stack(
            [expanded[name].reshape(-1) for name in offset_names], axis=0
        )
        observed = sums / counts + jnp.sum(
            reset_offsets[:, :, None] * reset_mask[:, None, :], axis=0
        )
        data_residual = (
            jnp.sqrt(1.0 / jnp.median(counts))
            / production.LOG_NOISE_SIGMA
            * jnp.sqrt(counts)
            * (prediction(values, curves)[None, :] - observed)
        ).reshape(-1)
        prior_residuals = [
            (
                expanded["initial_thickness_log_nm"]
                - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
            ).reshape(-1)
            / production.INITIAL_THICKNESS_LOG_SIGMA,
            *[
                (
                    expanded[name]
                    - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
                ).reshape(-1)
                / production.INITIAL_THICKNESS_LOG_SIGMA
                for name in thickness_names[1:]
            ],
            (
                values["constant_log_rate"]
                - np.log(PRODUCTION_TRAJECTORY_RATE_PRIOR_MEAN_NM_DAY)
            ).reshape(-1)
            / production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
            *[
                expanded[name].reshape(-1)
                / production.POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA
                for name in offset_names
            ],
        ]
        if "led_log_offsets" in expanded:
            prior_residuals.append(
                expanded["led_log_offsets"].reshape(-1)
                / led_log_offset_prior_sigma_log
            )
        return jnp.concatenate((data_residual, *prior_residuals))

    return {
        "residual": jax.jit(residual_function),
        "jacobian": jax.jit(jax.jacfwd(residual_function)),
        "unravel": unravel,
        "variable_site_names": variable_site_names,
        "rate_shape": rate_shape,
        "node_count": initial_rates.shape[-1],
    }


def fit_ice_only_constant_rate_reusable(
    data,
    cache,
    initial_values,
    evaluator,
    jax,
    jnp,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    """Fit one scalar-rate cached-response warmup with a reusable evaluator."""
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    initial_rates = np.asarray(initial_values["np_log_rate_nodes"], dtype=float)
    rate_shape = initial_rates.shape[:-1]
    variable_site_names = evaluator["variable_site_names"]
    variable_initial = {
        name: np.asarray(initial_values[name], dtype=float).copy()
        for name in variable_site_names
        if name != "constant_log_rate"
    }
    variable_initial["constant_log_rate"] = np.full(
        rate_shape,
        float(np.median(initial_rates)),
        dtype=float,
    )
    fixed_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in initial_values.items()
        if name not in variable_sites(initial_values)
    }
    flat_initial, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in variable_initial.items()}
    )
    parameter_scales = production.map_parameter_scale_values(
        initial_values,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    scale_values = {
        name: np.asarray(parameter_scales[name], dtype=float)
        for name in variable_site_names
        if name != "constant_log_rate"
    }
    scale_values["constant_log_rate"] = np.full(
        rate_shape,
        production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        dtype=float,
    )
    flat_scales, _ = ravel_pytree(
        {name: jnp.asarray(value) for name, value in scale_values.items()}
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)
    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    curves = jnp.asarray(cache["relative_log_response"])

    def scipy_residual(scaled_values):
        return np.asarray(
            evaluator["residual"](
                jnp.asarray(scaled_values),
                jnp.asarray(flat_initial),
                jnp.asarray(flat_scales),
                counts,
                sums,
                curves,
            ),
            dtype=float,
        )

    def scipy_jacobian(scaled_values):
        return np.asarray(
            evaluator["jacobian"](
                jnp.asarray(scaled_values),
                jnp.asarray(flat_initial),
                jnp.asarray(flat_scales),
                counts,
                sums,
                curves,
            ),
            dtype=float,
        )

    initial_scaled = np.zeros_like(flat_initial)
    started = time.perf_counter()
    initial_residual = scipy_residual(initial_scaled)
    compile_seconds = time.perf_counter() - started
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=production.PRODUCTION_MAP_STEPS,
        verbose=0,
    )
    total_seconds = time.perf_counter() - started
    variable_values = unravel(
        jnp.asarray(flat_initial) + jnp.asarray(flat_scales) * np.asarray(result.x)
    )
    values = dict(fixed_values)
    values.update(variable_values)
    constant_log_rate = values.pop("constant_log_rate")
    values["np_log_rate_nodes"] = jnp.broadcast_to(
        jnp.asarray(constant_log_rate)[..., None],
        rate_shape + (evaluator["node_count"],),
    )
    final_residual = scipy_residual(np.asarray(result.x, dtype=float))
    rate_nm_day = float(np.exp(np.asarray(values["np_log_rate_nodes"]).reshape(-1)[0]))
    return values, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameter_count": int(flat_initial.size),
        "compile_seconds": float(compile_seconds),
        "total_seconds": float(total_seconds),
        "residual_cost": float(0.5 * np.sum(final_residual**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_residual**2)),
        "constant_rate_map_rate_nm_day": rate_nm_day,
        "trajectory_rate_initialization_nm_day": float(
            np.exp(np.median(initial_rates))
        ),
    }


def build_ice_only_evaluator(
    data,
    initial_values,
    cache,
    jax,
    jnp,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    """Compile a tile-shaped residual/Jacobian that accepts dynamic tile data."""
    from jax.flatten_util import ravel_pytree

    variable_site_names = variable_sites(initial_values)
    thickness_names, offset_names = production.reset_site_names(data)
    template = {
        name: np.asarray(initial_values[name], dtype=float).copy()
        for name in variable_site_names
    }
    _, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in template.items()}
    )
    grid = jnp.asarray(np.asarray(cache["grid_nm"], dtype=float))
    days = jnp.asarray(data["days"])
    node_days = jnp.asarray(data["node_days"])
    led_index = jnp.asarray(np.asarray(data["led"], dtype=int) - 1, dtype=jnp.int32)
    first_by_led = jnp.asarray(data["first_by_led"], dtype=jnp.int32)
    group_index = jnp.arange(len(data["groups"]), dtype=jnp.int32)
    reset_mask = jnp.asarray(
        np.asarray(data["days"], dtype=float)[None, :]
        >= production.reset_days_for_data(data)[:, None]
    )
    layout_data = {
        "days": np.asarray(data["days"], dtype=float),
        "node_days": np.asarray(data["node_days"], dtype=float),
        "reset_day": float(data["reset_day"]),
        "reset_days": production.reset_days_for_data(data),
    }

    def prediction(values, curves):
        reset_initial = (
            jnp.stack([values[name] for name in thickness_names[1:]], axis=0)
            if len(thickness_names) > 2
            else values[thickness_names[1]]
        )
        ice, _, _, _ = production._trajectory_state(
            layout_data,
            values["np_log_rate_nodes"],
            values["initial_thickness_log_nm"],
            reset_initial,
            jax,
            jnp,
        )

        def interpolate_by_led(thickness):
            return jnp.stack(
                [jnp.interp(thickness, grid, curves[led]) for led in range(6)]
            )

        current_by_led = interpolate_by_led(ice[0])
        current = current_by_led[led_index, group_index]
        baseline_thickness = ice[0, first_by_led][led_index]
        baseline_by_led = interpolate_by_led(baseline_thickness)
        baseline = baseline_by_led[led_index, group_index]
        return current - baseline + _led_log_offset_values(data, values, jnp)[0]

    def residual_function(
        scaled_values,
        flat_initial,
        flat_scales,
        rate_reference,
        counts,
        sums,
        sum_squared,
        curves,
    ):
        values = unravel(flat_initial + flat_scales * scaled_values)
        reset_offsets = jnp.stack(
            [values[name].reshape(-1) for name in offset_names], axis=0
        )
        observed = sums / counts + jnp.sum(
            reset_offsets[:, :, None] * reset_mask[:, None, :], axis=0
        )
        data_residual = (
            jnp.sqrt(1.0 / jnp.median(counts))
            / production.LOG_NOISE_SIGMA
            * jnp.sqrt(counts)
            * (prediction(values, curves)[None, :] - observed)
        ).reshape(-1)
        prior_residuals = [
            (
                values["initial_thickness_log_nm"]
                - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
            ).reshape(-1)
            / production.INITIAL_THICKNESS_LOG_SIGMA,
            (
                values["np_log_rate_nodes"] - rate_reference
            ).reshape(-1)
            / production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        ]
        prior_residuals[1:1] = [
            (
                values[name]
                - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
            ).reshape(-1)
            / production.INITIAL_THICKNESS_LOG_SIGMA
            for name in thickness_names[1:]
        ]
        prior_residuals.extend(
            [
                values[name].reshape(-1)
                / production.POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA
                for name in offset_names
            ]
        )
        if "led_log_offsets" in values:
            prior_residuals.append(
                values["led_log_offsets"].reshape(-1)
                / led_log_offset_prior_sigma_log
            )
        return jnp.concatenate((data_residual, *prior_residuals))

    residual = jax.jit(residual_function)
    jacobian = jax.jit(jax.jacfwd(residual_function))
    return {"residual": residual, "jacobian": jacobian, "unravel": unravel}


def fit_ice_only_reusable(
    data,
    cache,
    initial_values,
    evaluator,
    jax,
    jnp,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    """Fit one tile using an evaluator compiled for reuse across same-shaped tiles."""
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    rate_reference = rate_prior_reference(initial_values)
    variable_site_names = variable_sites(initial_values)
    variable_initial = {
        name: np.asarray(initial_values[name], dtype=float).copy()
        for name in variable_site_names
    }
    fixed_values = {
        name: np.asarray(value, dtype=float).copy()
        for name, value in initial_values.items()
        if name not in variable_site_names
    }
    flat_initial, unravel = ravel_pytree(
        {name: jnp.asarray(value) for name, value in variable_initial.items()}
    )
    parameter_scales = production.map_parameter_scale_values(
        initial_values,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    flat_scales, _ = ravel_pytree(
        {name: jnp.asarray(parameter_scales[name]) for name in variable_site_names}
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)
    counts = jnp.asarray(data["counts"])
    sums = jnp.asarray(data["sums"])
    sum_squared = jnp.asarray(data["sum_squared"])
    curves = jnp.asarray(cache["relative_log_response"])

    def scipy_residual(scaled_values):
        return np.asarray(
            evaluator["residual"](
                jnp.asarray(scaled_values),
                jnp.asarray(flat_initial),
                jnp.asarray(flat_scales),
                jnp.asarray(rate_reference),
                counts,
                sums,
                sum_squared,
                curves,
            ),
            dtype=float,
        )

    def scipy_jacobian(scaled_values):
        return np.asarray(
            evaluator["jacobian"](
                jnp.asarray(scaled_values),
                jnp.asarray(flat_initial),
                jnp.asarray(flat_scales),
                jnp.asarray(rate_reference),
                counts,
                sums,
                sum_squared,
                curves,
            ),
            dtype=float,
        )

    initial_scaled = np.zeros_like(flat_initial)
    started = time.perf_counter()
    initial_residual = scipy_residual(initial_scaled)
    compile_seconds = time.perf_counter() - started
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=production.PRODUCTION_MAP_STEPS,
        verbose=0,
    )
    total_seconds = time.perf_counter() - started
    values = dict(fixed_values)
    values.update(
        unravel(
            jnp.asarray(flat_initial)
            + jnp.asarray(flat_scales) * np.asarray(result.x, dtype=float)
        )
    )
    final_residual = scipy_residual(np.asarray(result.x, dtype=float))
    return values, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameter_count": int(flat_initial.size),
        "compile_seconds": float(compile_seconds),
        "total_seconds": float(total_seconds),
        "residual_cost": float(0.5 * np.sum(final_residual**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_residual**2)),
        "trajectory_rate_initialization_nm_day": float(
            np.exp(np.median(variable_initial["np_log_rate_nodes"]))
        ),
    }


def fit_ice_only_batch(
    data_list,
    cache_list,
    initial_values_list,
    jax,
    jnp,
    led_log_offset_prior_sigma_log=production.LED_LOG_OFFSET_PRIOR_SIGMA_LOG,
):
    """Fit independent tiles together while batching JAX residual evaluations."""
    data = _stack_tile_data(data_list)
    variable_site_names = variable_sites(initial_values_list[0])
    thickness_names, offset_names = production.reset_site_names(data)
    grids = [np.asarray(cache["grid_nm"], dtype=float) for cache in cache_list]
    if not all(np.array_equal(grids[0], grid) for grid in grids[1:]):
        raise ValueError("Cannot batch response caches with different thickness grids")
    batch_cache = {
        "grid_nm": grids[0],
        "relative_log_response": np.stack(
            [np.asarray(cache["relative_log_response"], dtype=float) for cache in cache_list]
        ),
    }
    variable_initial = {}
    for name in variable_site_names:
        if name == "led_log_offsets":
            variable_initial[name] = np.concatenate(
                [
                    np.asarray(values[name], dtype=float).reshape(
                        (-1, np.asarray(values[name]).shape[-1])
                    )
                    for values in initial_values_list
                ],
                axis=0,
            )
        else:
            variable_initial[name] = np.concatenate(
                [
                    np.asarray(values[name], dtype=float).reshape(
                        (-1,) + np.asarray(values[name]).shape[2:]
                    )
                    for values in initial_values_list
                ],
                axis=0,
            )
    tile_count = len(initial_values_list)
    rate_reference = rate_prior_reference(variable_initial)
    initial_rate_nm_day = np.exp(
        np.median(variable_initial["np_log_rate_nodes"], axis=1)
    )

    from jax.flatten_util import ravel_pytree
    from scipy.optimize import least_squares

    flat_initial, unravel = ravel_pytree(
        {name: jnp.asarray(variable_initial[name]) for name in variable_site_names}
    )
    parameter_scales = production.map_parameter_scale_values(
        variable_initial,
        led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
    )
    flat_scales, _ = ravel_pytree(
        {name: jnp.asarray(parameter_scales[name]) for name in variable_site_names}
    )
    flat_initial = np.asarray(flat_initial, dtype=float)
    flat_scales = np.asarray(flat_scales, dtype=float)

    def full_values(scaled_values):
        return unravel(
            jnp.asarray(flat_initial) + jnp.asarray(flat_scales) * scaled_values
        )

    likelihood_weights = 1.0 / np.median(
        np.asarray(data["counts"], dtype=float), axis=1
    )
    likelihood_weights = jnp.asarray(likelihood_weights)
    reset_mask = jnp.asarray(
        np.asarray(data["days"], dtype=float)[None, :]
        >= production.reset_days_for_data(data)[:, None]
    )
    counts = jnp.asarray(data["counts"])

    def residual_matrix(scaled_values):
        values = full_values(scaled_values)
        state = relative_state_batch(data, values, batch_cache, jax, jnp)
        observed = jnp.asarray(data["sums"]) / counts
        reset_offsets = jnp.stack(
            [values[name].reshape(-1) for name in offset_names], axis=0
        )
        observed = observed + jnp.sum(
            reset_offsets[:, :, None] * reset_mask[:, None, :], axis=0
        )
        data_residual = (
            jnp.sqrt(likelihood_weights[:, None])
            / production.LOG_NOISE_SIGMA
            * jnp.sqrt(counts)
            * (state["prediction"] - observed)
        )
        initial_residual = (
            values["initial_thickness_log_nm"]
            - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
        ).reshape(-1, 1) / production.INITIAL_THICKNESS_LOG_SIGMA
        reset_initial_residual = jnp.concatenate(
            [
                (
                    values[name]
                    - np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
                ).reshape(-1, 1)
                / production.INITIAL_THICKNESS_LOG_SIGMA
                for name in thickness_names[1:]
            ],
            axis=1,
        )
        rate_residual = (
            values["np_log_rate_nodes"] - jnp.asarray(rate_reference)
        ) / production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG
        offset_residual = jnp.concatenate(
            [
                values[name].reshape(-1, 1)
                / production.POST_RESET_TILE_LOG_OFFSET_PRIOR_SIGMA
                for name in offset_names
            ],
            axis=1,
        )
        if "led_log_offsets" in values:
            led_offset_residual = values["led_log_offsets"] / led_log_offset_prior_sigma_log
        else:
            led_offset_residual = jnp.zeros(
                (tile_count, 0),
                dtype=data_residual.dtype,
            )
        return jnp.concatenate(
            (
                data_residual,
                initial_residual,
                reset_initial_residual,
                rate_residual,
                offset_residual,
                led_offset_residual,
            ),
            axis=1,
        )

    residual_function = lambda scaled: residual_matrix(scaled).reshape(-1)
    residual_jit = jax.jit(residual_function)
    jacobian_jit = jax.jit(jax.jacfwd(residual_function))

    def scipy_residual(scaled_values):
        return np.asarray(residual_jit(jnp.asarray(scaled_values)), dtype=float)

    def scipy_jacobian(scaled_values):
        return np.asarray(jacobian_jit(jnp.asarray(scaled_values)), dtype=float)

    initial_scaled = np.zeros_like(flat_initial)
    started = time.perf_counter()
    initial_matrix = np.asarray(
        residual_matrix(jnp.asarray(initial_scaled)), dtype=float
    )
    compile_seconds = time.perf_counter() - started
    result = least_squares(
        scipy_residual,
        initial_scaled,
        jac=scipy_jacobian,
        method="trf",
        x_scale=1.0,
        ftol=production.PRODUCTION_MAP_FTOL,
        xtol=production.PRODUCTION_MAP_XTOL,
        gtol=1.0e-9,
        max_nfev=production.PRODUCTION_MAP_STEPS,
        verbose=0,
    )
    total_seconds = time.perf_counter() - started
    values = full_values(np.asarray(result.x, dtype=float))
    final_matrix = np.asarray(
        residual_matrix(np.asarray(result.x, dtype=float)), dtype=float
    )
    return values, {
        "converged": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "parameter_count": int(flat_initial.size // tile_count),
        "compile_seconds": float(compile_seconds),
        "total_seconds": float(total_seconds),
        "residual_cost": float(0.5 * np.sum(final_matrix**2)),
        "initial_residual_cost": float(0.5 * np.sum(initial_matrix**2)),
        "initial_residual_costs": 0.5 * np.sum(initial_matrix**2, axis=1),
        "residual_costs": 0.5 * np.sum(final_matrix**2, axis=1),
        "trajectory_rate_initialization_nm_day": initial_rate_nm_day,
        "tile_count": tile_count,
    }


def build_spectral_response_lookup(
    optical_values,
    wavelengths_nm,
    ice_thickness_nm,
    incidence_pupil_half_angle_deg=0.0,
    complex_reflection_modulation_ratio=None,
):
    """Evaluate fixed-optics response before LED/SED/QE integration.

    The returned relative response is the log of coated/clean power at each
    wavelength and ice thickness.  The clean spectrum is retained so a later
    consumer can apply arbitrary filter, SED, and detector-QE weights.
    """
    jax, jnp = production.require_jax()
    jax.config.update("jax_enable_x64", True)
    wavelengths = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    thickness = np.asarray(ice_thickness_nm, dtype=float).reshape(-1)
    if (
        wavelengths.size < 2
        or np.any(~np.isfinite(wavelengths))
        or np.any(np.diff(wavelengths) <= 0.0)
    ):
        raise ValueError("wavelengths_nm must be finite and strictly increasing")
    if thickness.size < 1 or np.any(~np.isfinite(thickness)) or np.any(thickness < 0.0):
        raise ValueError("ice_thickness_nm must be finite and non-negative")
    if complex_reflection_modulation_ratio is None:
        modulation_ratio = None
    else:
        modulation_ratio = np.asarray(
            complex_reflection_modulation_ratio, dtype=complex
        ).reshape(-1)
        if (
            modulation_ratio.shape != wavelengths.shape
            or np.any(~np.isfinite(modulation_ratio))
            or np.any(np.abs(modulation_ratio) <= 0.0)
        ):
            raise ValueError(
                "complex_reflection_modulation_ratio must be finite, nonzero, "
                "and match the wavelength grid"
            )
        modulation_ratio = jnp.asarray(modulation_ratio)

    pupil_directions, pupil_weights = production.incidence_pupil_quadrature(
        incidence_pupil_half_angle_deg
    )
    pupil_mode = float(incidence_pupil_half_angle_deg) > 0.0
    pupil_local_x = jnp.asarray(pupil_directions[:, 0])
    pupil_local_y = jnp.asarray(pupil_directions[:, 1])
    pupil_local_z = jnp.asarray(pupil_directions[:, 2])
    pupil_weights = jnp.asarray(pupil_weights)
    pupil_count = int(pupil_directions.shape[0])
    wavelengths_jnp = jnp.asarray(wavelengths)
    phase_factor = 2.0 * jnp.pi / wavelengths_jnp

    ar_raw = np.asarray(optical_values["ar_tile_raw"], dtype=float).reshape(-1)
    ar_physical = np.asarray(
        production._physical_ar_parameters(jnp.asarray(ar_raw[None, :]), jax, jnp),
        dtype=float,
    ).reshape(1, production.AR_LAYERS, 3)
    base_n, base_k = production.hgcdte_reference_optical_constants(wavelengths)
    hg_n = base_n * np.exp(
        float(np.asarray(optical_values["absolute_hgcdte_n_log_scale"]).reshape(-1)[0])
    )
    hg_k = base_k * np.exp(
        float(np.asarray(optical_values["absolute_hgcdte_k_log_scale"]).reshape(-1)[0])
    )
    angle_deg = float(np.asarray(optical_values["incidence_angle_deg"]).reshape(-1)[0])

    if pupil_mode:
        angle_rad = jnp.asarray(angle_deg) * jnp.pi / 180.0
        sine_angle = jnp.sin(angle_rad)
        cosine_angle = jnp.cos(angle_rad)
        direction_x = pupil_local_x * cosine_angle + pupil_local_z * sine_angle
        direction_y = pupil_local_y
        pupil_angle_sine = jnp.sqrt(
            jnp.maximum(direction_x * direction_x + direction_y * direction_y, 0.0)
        )
        response_angle_sine = jnp.tile(pupil_angle_sine, thickness.size)
        response_ar = jnp.repeat(
            jnp.asarray(ar_physical.reshape(1, -1)),
            thickness.size * pupil_count,
            axis=0,
        )
        response_hg_n = jnp.repeat(
            jnp.asarray(hg_n[None, :]), thickness.size * pupil_count, axis=0
        )
        response_hg_k = jnp.repeat(
            jnp.asarray(hg_k[None, :]), thickness.size * pupil_count, axis=0
        )
        response_thickness = jnp.repeat(jnp.asarray(thickness), pupil_count)
    else:
        response_count = thickness.size
        response_angle_sine = jnp.full(response_count, production._incidence_sine(angle_deg, jnp))
        response_ar = jnp.broadcast_to(
            jnp.asarray(ar_physical.reshape(1, -1)), (response_count, 9)
        )
        response_hg_n = jnp.broadcast_to(
            jnp.asarray(hg_n[None, :]), (response_count, wavelengths.size)
        )
        response_hg_k = jnp.broadcast_to(
            jnp.asarray(hg_k[None, :]), (response_count, wavelengths.size)
        )
        response_thickness = jnp.asarray(thickness)

    def polarization_response(ar_parameters, hgcdte_n, hgcdte_k, polarization):
        ar_parameters = ar_parameters.reshape((ar_parameters.shape[0], 3, 3))
        ar_index = ar_parameters[:, :, 1] + 1j * ar_parameters[:, :, 2]
        ar_thickness = 100.0 * ar_parameters[:, :, 0]
        substrate = hgcdte_n + 1j * hgcdte_k
        substrate_cosine = jnp.sqrt(
            1.0 - (response_angle_sine[:, None] / substrate) ** 2
        )
        if polarization == "s":
            substrate_admittance = substrate * substrate_cosine
        else:
            substrate_admittance = substrate / substrate_cosine
        incident_cosine = jnp.sqrt(1.0 - response_angle_sine**2)
        if polarization == "s":
            incident_admittance = incident_cosine
        else:
            incident_admittance = 1.0 / incident_cosine

        boundary_first = jnp.ones(
            (ar_parameters.shape[0], wavelengths.size), dtype=jnp.complex128
        )
        boundary_second = substrate_admittance
        for layer in range(production.AR_LAYERS - 1, -1, -1):
            index = ar_index[:, layer]
            cosine = jnp.sqrt(
                1.0 - (response_angle_sine / index) ** 2
            )
            if polarization == "s":
                admittance = index * cosine
            else:
                admittance = index / cosine
            phase = (
                phase_factor[None, :]
                * index[:, None]
                * ar_thickness[:, layer, None]
                * cosine[:, None]
            )
            cosine_phase = jnp.cos(phase)
            sine_phase = -1j * jnp.sin(phase)
            old_first, old_second = boundary_first, boundary_second
            boundary_first = cosine_phase * old_first + sine_phase / admittance[:, None] * old_second
            boundary_second = sine_phase * admittance[:, None] * old_first + cosine_phase * old_second
        coated_boundary_second = boundary_second
        clean_denominator = incident_admittance[:, None] * boundary_first + boundary_second
        if modulation_ratio is not None:
            clean_reflection = (
                incident_admittance[:, None] * boundary_first - boundary_second
            ) / clean_denominator
            modulated_clean_reflection = (
                clean_reflection * modulation_ratio[None, :]
            )
            clean_input_admittance = incident_admittance[:, None] * (
                1.0 - modulated_clean_reflection
            ) / (1.0 + modulated_clean_reflection)
            coated_boundary_second = boundary_first * clean_input_admittance
            clean_denominator = (
                incident_admittance[:, None] * boundary_first
                + coated_boundary_second
            )
        clean = (
            jnp.real(substrate_admittance / incident_admittance[:, None])
            * jnp.abs(2.0 * incident_admittance[:, None] / clean_denominator) ** 2
        )

        ice_index = jnp.asarray(production.ICE_INDEX_DEFAULT + 0.0j)
        ice_cosine = jnp.sqrt(
            1.0 - (response_angle_sine[:, None] / ice_index) ** 2
        )
        if polarization == "s":
            ice_admittance = ice_index * ice_cosine
        else:
            ice_admittance = ice_index / ice_cosine
        ice_phase = (
            phase_factor[None, :]
            * response_thickness[:, None]
            * ice_index
            * ice_cosine
        )
        ice_cosine_phase = jnp.cos(ice_phase)
        ice_sine_phase = -1j * jnp.sin(ice_phase)
        coated_first = (
            ice_cosine_phase * boundary_first
            + ice_sine_phase / ice_admittance * coated_boundary_second
        )
        coated_second = (
            ice_sine_phase * ice_admittance * boundary_first
            + ice_cosine_phase * coated_boundary_second
        )
        coated_denominator = (
            incident_admittance[:, None] * coated_first + coated_second
        )
        coated = (
            jnp.real(substrate_admittance / incident_admittance[:, None])
            * jnp.abs(2.0 * incident_admittance[:, None] / coated_denominator) ** 2
        )
        return clean, coated

    clean_s, coated_s = polarization_response(
        response_ar, response_hg_n, response_hg_k, "s"
    )
    clean_p, coated_p = polarization_response(
        response_ar, response_hg_n, response_hg_k, "p"
    )
    clean = 0.5 * (clean_s + clean_p)
    coated = 0.5 * (coated_s + coated_p)
    if pupil_mode:
        clean = jnp.sum(
            clean.reshape((thickness.size, pupil_count, wavelengths.size))
            * pupil_weights[None, :, None],
            axis=1,
        )
        coated = jnp.sum(
            coated.reshape((thickness.size, pupil_count, wavelengths.size))
            * pupil_weights[None, :, None],
            axis=1,
        )
    clean_reference = clean[0]
    return {
        "wavelengths_nm": wavelengths,
        "ice_thickness_nm": thickness,
        "clean_power": np.asarray(clean_reference, dtype=float),
        "relative_log_response": np.asarray(
            jnp.log(coated / clean_reference[None, :]), dtype=float
        ),
    }


def write_spectral_response_cache(
    path,
    detector,
    tile_id,
    spectral_response,
    source_fit,
    rank,
):
    """Write a low-rank wavelength/thickness response cache."""
    wavelengths = np.asarray(spectral_response["wavelengths_nm"], dtype=float)
    thickness = np.asarray(spectral_response["ice_thickness_nm"], dtype=float)
    clean_power = np.asarray(spectral_response["clean_power"], dtype=float)
    relative = np.asarray(spectral_response["relative_log_response"], dtype=float)
    if relative.shape != (thickness.size, wavelengths.size):
        raise ValueError("Spectral response shape does not match its grids")
    if not 1 <= int(rank) <= min(relative.shape):
        raise ValueError("Spectral cache rank is outside the response dimensions")
    left, singular_values, right = np.linalg.svd(relative, full_matrices=False)
    left = left[:, :rank] * singular_values[:rank]
    right = right[:rank]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.asarray(SPECTRAL_CACHE_VERSION),
        detector=np.asarray(str(detector)),
        tile_id=np.asarray(int(tile_id)),
        wavelengths_nm=wavelengths.astype(np.float32),
        ice_thickness_nm=thickness.astype(np.float32),
        clean_power=clean_power.astype(np.float32),
        response_left=left.astype(np.float32),
        response_right=right.astype(np.float32),
        response_rank=np.asarray(int(rank)),
        response_representation=np.asarray("low_rank_linear_grid"),
        source_fit=np.asarray(portable_path_string(source_fit)),
    )
    return path


def load_spectral_response_cache(path, detector=None, tile_id=None):
    """Load and validate a low-rank wavelength/thickness cache."""
    with np.load(Path(path), allow_pickle=False) as source:
        version = int(np.asarray(source["cache_version"]).reshape(-1)[0])
        if version != SPECTRAL_CACHE_VERSION:
            raise ValueError(f"Unsupported spectral response cache version {version}")
        cached_detector = str(np.asarray(source["detector"]).reshape(-1)[0])
        cached_tile = int(np.asarray(source["tile_id"]).reshape(-1)[0])
        if detector is not None and cached_detector != str(detector):
            raise ValueError("Spectral response cache detector does not match")
        if tile_id is not None and cached_tile != int(tile_id):
            raise ValueError("Spectral response cache tile does not match")
        return {
            "detector": cached_detector,
            "tile_id": cached_tile,
            "wavelengths_nm": np.asarray(source["wavelengths_nm"], dtype=float),
            "ice_thickness_nm": np.asarray(source["ice_thickness_nm"], dtype=float),
            "clean_power": np.asarray(source["clean_power"], dtype=float),
            "response_left": np.asarray(source["response_left"], dtype=float),
            "response_right": np.asarray(source["response_right"], dtype=float),
            "source_fit": str(
                resolve_portable_path(
                    str(np.asarray(source["source_fit"]).reshape(-1)[0])
                )
            ),
        }


def evaluate_spectral_response_cache(cache, ice_thickness_nm, wavelengths_nm):
    """Evaluate a low-rank cache with linear interpolation in both axes."""
    source_thickness = np.asarray(cache["ice_thickness_nm"], dtype=float)
    source_wavelengths = np.asarray(cache["wavelengths_nm"], dtype=float)
    thickness = np.asarray(ice_thickness_nm, dtype=float).reshape(-1)
    wavelengths = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    if (
        np.any(~np.isfinite(thickness))
        or np.any(thickness < source_thickness[0])
        or np.any(thickness > source_thickness[-1])
    ):
        raise ValueError("ice thickness is outside the spectral cache grid")
    if (
        np.any(~np.isfinite(wavelengths))
        or np.any(wavelengths < source_wavelengths[0])
        or np.any(wavelengths > source_wavelengths[-1])
    ):
        raise ValueError("wavelength is outside the spectral cache grid")
    left = np.asarray(cache["response_left"], dtype=float)
    right = np.asarray(cache["response_right"], dtype=float)
    interpolated_left = np.stack(
        [np.interp(thickness, source_thickness, left[:, rank]) for rank in range(left.shape[1])],
        axis=1,
    )
    interpolated_right = np.stack(
        [np.interp(wavelengths, source_wavelengths, right[rank]) for rank in range(right.shape[0])],
        axis=0,
    )
    relative_log_response = interpolated_left @ interpolated_right
    clean_power = np.interp(wavelengths, source_wavelengths, cache["clean_power"])
    return clean_power, relative_log_response


def integrate_spectral_response_cache(cache, ice_thickness_nm, wavelengths_nm, weights):
    """Integrate cached clean/coated spectra against one or more filter weights."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.ndim != 2 or weights.shape[1] != wavelengths.size:
        raise ValueError("weights must have shape (filter_count, wavelength_count)")
    clean_power, relative_log_response = evaluate_spectral_response_cache(
        cache, ice_thickness_nm, wavelengths
    )
    clean_integral = np.trapezoid(weights * clean_power[None, :], wavelengths, axis=1)
    coated_integral = np.trapezoid(
        weights[None, :, :] * clean_power[None, None, :]
        * np.exp(relative_log_response[:, None, :]),
        wavelengths,
        axis=2,
    )
    if np.any(clean_integral <= 0.0) or np.any(coated_integral <= 0.0):
        raise ValueError("filter weights must produce positive integrated power")
    return np.log(coated_integral / clean_integral[None, :])
