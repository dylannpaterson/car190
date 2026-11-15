"""Fast H5-only, batched independent MAP update of WFI ice histories.

The unchanged Eric core evaluates each tile's six-LED response from the H5.
Corrected statistics then fit independent 48-hour log-rate splines with knots
at resets, plus initial/reset thicknesses and LED offsets using the production
likelihood and priors. No TMM or CAR190/Amber joint optimizer is run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
from astropy.time import Time

import fit_wfi_tmm_production as production
import fit_wfi_relative_response as relative_fit
import compare_eric_v3_amber_car201 as reader_loader
import run_wfi_relative_response_production as relative_runner
import wfi_relative_response as relative_model
import wfi_wavelength_response_bundle as bundle_io
from export_joint_ice_history_reference import deposition_factorizations_on_blocks
from wfi_pipeline_defaults import (
    DETECTORS,
    RELATIVE_RESPONSE,
    THIRD_DECON_DAY,
    TILE_SIZE_PIXELS,
    output_paths,
    pinned_relative_response_cache,
)
from wfi_production_layout import (
    complete_tile_ids,
    corrected_statistics_path,
    tile_statistics_path,
)
from wfi_relative_response import portable_path_string


DEFAULT_LOOKUP_CACHE = "ice_history_lookup"
DEFAULT_RATE_MEAN_NM_DAY = 12.0
DEFAULT_RATE_SIGMA_LOG = 1.5
DEFAULT_LOOKUP_STEP_NM = 2.0
DEFAULT_HISTORY_BRANCH_SIGMA_NM = 1.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def h5_optical_fingerprint(model):
    """Fingerprint fixed H5 optics without hashing its changing deposition history."""
    digest = hashlib.sha256()
    for name, value in (
        ("wavelength_nm", model.wavelength_nm),
        ("mean", model._mean),
        ("modes", model._modes),
        ("amplitudes", model._amp),
        ("cos_theta", model._cos_theta),
        ("film_n", model._n),
        ("film_kappa", model._kappa),
        ("film_b", model._b),
        ("led_nodes_nm", model.led_nodes_nm),
        ("led_weight", model.led_weight),
    ):
        array = np.asarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def find_bundle_root(base_h5, explicit=None):
    if explicit is not None:
        root = Path(explicit).resolve()
        if not (root / "wfi_all_2nm_rank16_modulated.npz").is_file():
            raise FileNotFoundError(
                root / "wfi_all_2nm_rank16_modulated.npz"
            )
        return root
    for parent in Path(base_h5).resolve().parents:
        if (parent / "wfi_all_2nm_rank16_modulated.npz").is_file():
            return parent
    raise FileNotFoundError(
        "Could not find wfi_all_2nm_rank16_modulated.npz beside the base H5; "
        "pass --bundle-root"
    )


def find_base_run_root(base_h5, bundle_root, explicit=None):
    if explicit is not None:
        return Path(explicit).resolve()
    candidates = [bundle_root.parent.parent]
    candidates.extend(Path(base_h5).resolve().parents)
    for root in candidates:
        expected = (
            root
            / "production"
            / "wfi01"
            / "relative_response"
            / "fits"
            / "tile_000"
            / "posterior_samples.npz"
        )
        if expected.is_file():
            return root
    raise FileNotFoundError(
        "Could not locate the base run's per-tile fits, which contain the saved "
        "LED offsets and prior full histories; pass --base-run-root"
    )


def tile_superpixel_indices(model, detector, tile_id, tile_size=TILE_SIZE_PIXELS):
    """Map one 256-pixel statistics tile to its Eric-reader superpixels."""
    sca = int(str(detector).upper()[3:])
    cells_per_tile = int(tile_size) // int(model.superpixel_px)
    if cells_per_tile < 1 or int(tile_size) % int(model.superpixel_px):
        raise ValueError("tile size must be divisible by the H5 superpixel width")
    detector_tiles = 4096 // int(tile_size)
    tile_row, tile_col = divmod(int(tile_id), detector_tiles)
    rows = np.asarray(model._row, dtype=int)
    cols = np.asarray(model._col, dtype=int)
    scas = np.asarray(model._sca, dtype=int)
    selected = np.flatnonzero(
        (scas == sca)
        & (rows // cells_per_tile == tile_row)
        & (cols // cells_per_tile == tile_col)
    )
    expected = cells_per_tile**2
    if selected.size != expected:
        raise ValueError(
            f"{detector} tile {tile_id}: H5 maps {selected.size} superpixels, "
            f"expected {expected}"
        )
    return selected


def build_tile_led_lookup(
    model,
    core,
    detector,
    tile_id,
    thickness_nm,
    *,
    thickness_chunk=64,
    uniform_tolerance=1.0e-7,
):
    """Build mean per-pixel log LED response versus physical ice thickness.

    The H5's spatial basis is constant inside a 256-pixel TMM tile in the
    standard product. If a future H5 has sub-tile variation, this function
    averages the exact per-superpixel log response instead of discarding it.
    """
    thickness_nm = np.asarray(thickness_nm, dtype=float).reshape(-1)
    if (
        thickness_nm.size < 2
        or np.any(~np.isfinite(thickness_nm))
        or np.any(np.diff(thickness_nm) <= 0.0)
    ):
        raise ValueError("lookup thickness grid must be finite and increasing")
    if model.led_nodes_nm is None or model.led_weight is None:
        raise ValueError("Base H5 has no six-LED quadrature")

    superpixels = tile_superpixel_indices(model, detector, tile_id)
    amplitudes = np.asarray(model._amp[superpixels])
    cos_theta = np.asarray(model._cos_theta[superpixels], dtype=float)
    representative = np.asarray([0], dtype=int)
    if (
        np.max(np.abs(amplitudes - amplitudes[:1])) > uniform_tolerance
        or np.max(np.abs(cos_theta - cos_theta[:1])) > uniform_tolerance
    ):
        representative = np.arange(superpixels.size, dtype=int)
    superpixels = superpixels[representative]
    amplitudes = amplitudes[representative]
    cos_theta = cos_theta[representative]

    led_nodes = np.asarray(model.led_nodes_nm, dtype=float)
    led_weights = np.asarray(model.led_weight, dtype=float)
    if led_nodes.ndim != 2 or led_weights.shape != led_nodes.shape:
        raise ValueError("Base H5 six-LED wavelength/weight arrays are inconsistent")
    n_led, n_node = led_nodes.shape
    wavelength_nm = led_nodes.reshape(-1)
    wavelength_count = wavelength_nm.size
    reflectance = core.reflectance(
        amplitudes,
        np.asarray(model.wavelength_nm, dtype=float),
        np.asarray(model._mean),
        np.asarray(model._modes),
        wavelength_nm,
    )
    n_l, kappa_l, b_l = core.optics_at(
        model.wavelength_nm,
        model._n,
        model._kappa,
        model._b,
        wavelength_nm,
    )
    no_ice = np.asarray(core.no_ice_transmission(reflectance, b_l), dtype=float)
    if no_ice.shape != (superpixels.size, wavelength_count):
        raise ValueError(f"{detector}/tile {tile_id}: invalid H5 no-ice shape")

    weighted_no_ice = (
        no_ice.reshape(superpixels.size, n_led, n_node)
        * led_weights[None, :, :]
    )
    denominator = weighted_no_ice.sum(axis=2)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError(f"{detector}/tile {tile_id}: invalid LED normalization")

    result = np.empty((n_led, thickness_nm.size), dtype=float)
    for start in range(0, thickness_nm.size, int(thickness_chunk)):
        stop = min(start + int(thickness_chunk), thickness_nm.size)
        grid = thickness_nm[start:stop]
        d_phys = np.broadcast_to(grid[None, :], (superpixels.size, grid.size))
        response = core.relative_response(
            reflectance,
            wavelength_nm,
            d_phys,
            cos_theta,
            model.wavelength_nm,
            model._n,
            model._kappa,
            model._b,
            model._n_ref,
            model._snell,
        )
        response = response.reshape(superpixels.size, n_led, n_node, grid.size)
        weights = weighted_no_ice[:, :, :, None]
        per_cell_ratio = np.sum(response * weights, axis=2) / denominator[:, :, None]
        if np.any(~np.isfinite(per_cell_ratio)) or np.any(per_cell_ratio <= 0.0):
            raise ValueError(f"{detector}/tile {tile_id}: invalid H5 LED response lookup")
        result[:, start:stop] = np.mean(np.log(per_cell_ratio), axis=0)
    if np.max(np.abs(result[:, thickness_nm == 0.0])) > 1.0e-8:
        raise ValueError(f"{detector}/tile {tile_id}: zero-thickness response is not unity")
    return result


def _piecewise_interp(source_days, source_values, query_days, reset_days):
    source_days = np.asarray(source_days, dtype=float).reshape(-1)
    source_values = np.asarray(source_values, dtype=float).reshape(-1)
    query_days = np.asarray(query_days, dtype=float).reshape(-1)
    if source_days.size != source_values.size or source_days.size == 0:
        raise ValueError("prior history has an invalid time/thickness axis")
    source_segment = np.searchsorted(reset_days, source_days, side="right")
    query_segment = np.searchsorted(reset_days, query_days, side="right")
    output = np.empty_like(query_days)
    for segment in np.unique(query_segment):
        target = query_segment == segment
        source = source_segment == segment
        if not np.any(source):
            raise ValueError(f"Prior history has no samples in reset segment {segment}")
        output[target] = np.interp(
            query_days[target], source_days[source], source_values[source]
        )
    return output


def _base_fit_path(base_run_root, detector, tile_id):
    path = (
        Path(base_run_root)
        / "production"
        / detector.lower()
        / "relative_response"
        / "fits"
        / f"tile_{int(tile_id):03d}"
        / "posterior_samples.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Base per-tile fit is missing: {path}")
    return path


def _fit_seed(archive):
    led = np.asarray(archive.get("led_log_offsets", np.zeros((1, 6))), dtype=float).reshape(-1)
    if led.size != 6:
        raise ValueError("Base fit LED-offset vector must have six entries")
    reset = float(np.asarray(archive.get("post_reset_tile_log_offset", 0.0)).reshape(-1)[0])
    third = float(np.asarray(archive.get("post_third_decon_tile_log_offset", 0.0)).reshape(-1)[0])
    days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
    thickness = np.asarray(archive["tile_ice_thickness"], dtype=float).reshape(-1)
    if days.size != thickness.size or days.size == 0:
        raise ValueError("Base fit history has an invalid observation axis")
    return led, reset, third, days, thickness


def _initial_thickness_prior(base_archive, prior_name):
    """Recover the production prior, rather than treating its MAP as a prior."""
    prior_stages = str(
        np.asarray(base_archive.get("thickness_prior_stages", ""))
        .reshape(-1)[0]
    )
    applies_to_stage = prior_stages == "all" or (
        prior_stages == "third-decon"
        and prior_name == "third_decon_initial_thickness_log_nm"
    )
    if not applies_to_stage:
        return (
            float(np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)),
            float(production.INITIAL_THICKNESS_LOG_SIGMA),
        )

    archived_root = base_archive.get("warmup_prior_root")
    if archived_root is None:
        raise ValueError(
            f"The base fit marks {prior_stages!r} warmup thickness priors but has no warmup_prior_root"
        )
    tile_id = int(np.asarray(base_archive["tile_ids"]).reshape(-1)[0])
    warmup_root = relative_fit.resolve_portable_path(
        str(np.asarray(archived_root).reshape(-1)[0])
    )
    warmup_path = (
        warmup_root / "fits" / f"tile_{tile_id:03d}" / "constant_rate_map.npz"
    )
    if not warmup_path.is_file():
        raise FileNotFoundError(
            f"Missing warmup thickness-prior archive for tile {tile_id}: {warmup_path}"
        )
    with np.load(warmup_path, allow_pickle=False) as warmup:
        if prior_name not in warmup.files:
            raise ValueError(f"Warmup archive {warmup_path} lacks {prior_name}")
        center = float(np.asarray(warmup[prior_name], dtype=float).reshape(-1)[0])
    sigma = float(
        np.asarray(base_archive["thickness_prior_log_sigma"], dtype=float)
        .reshape(-1)[0]
    )
    if not np.isfinite(center) or not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"Invalid warmup thickness prior in {warmup_path}")
    return center, sigma


def lookup_tile_history(
    data,
    lookup_log_response,
    thickness_grid,
    base_archive,
    seed_history_days=None,
    seed_history_thickness=None,
    branch_selection_sigma_nm=DEFAULT_HISTORY_BRANCH_SIGMA_NM,
):
    """Invert all six-LED repeat nodes, using the prior history to choose branches."""
    grid = np.asarray(thickness_grid, dtype=float).reshape(-1)
    curves = np.asarray(lookup_log_response, dtype=float)
    if curves.shape != (6, grid.size):
        raise ValueError("H5 LED lookup has an invalid shape")
    led_offsets, reset_offset, third_offset, base_days, base_ice = _fit_seed(base_archive)
    reset_days = production.reset_days_for_data(data)
    days = np.asarray(data["days"], dtype=float)
    repeat_values = np.asarray(data["repeat_values"], dtype=int)
    node_days = np.asarray(data["node_days"], dtype=float)
    row_repeat = np.asarray(data["repeat"], dtype=int)
    row_led = np.asarray(data["led"], dtype=int) - 1
    counts = np.asarray(data["counts"], dtype=float).reshape(-1)
    sums = np.asarray(data["sums"], dtype=float).reshape(-1)
    if np.any((row_led < 0) | (row_led >= 6)):
        raise ValueError("Statistics contain an LED outside the H5 six-LED layout")

    base_at_rows = _piecewise_interp(base_days, base_ice, days, reset_days)
    if seed_history_days is None:
        seed_history_days = base_days
        seed_history_thickness = base_ice
    seed_history_days = np.asarray(seed_history_days, dtype=float).reshape(-1)
    seed_history_thickness = np.asarray(seed_history_thickness, dtype=float).reshape(-1)
    seed_at_nodes = _piecewise_interp(
        seed_history_days, seed_history_thickness, node_days, reset_days
    )
    reference_indices = np.asarray(data["first_by_led"], dtype=int)
    reference_rows = np.zeros(counts.size, dtype=bool)
    reference_rows[reference_indices] = True
    base_led_thickness = base_at_rows[reference_indices]
    base_led_response = np.asarray(
        [np.interp(base_led_thickness[led], grid, curves[led]) for led in range(6)]
    )
    median_count = max(float(np.median(counts)), 1.0)
    noise_sigma = float(production.LOG_NOISE_SIGMA)
    update_records = []
    node_history = np.empty_like(node_days, dtype=float)
    previous_thickness = None
    previous_day = None
    previous_episode = None

    for node_index, (repeat, node_day) in enumerate(zip(repeat_values, node_days, strict=True)):
        rows = np.flatnonzero(row_repeat == repeat)
        if not rows.size:
            continue
        episode = int(np.searchsorted(reset_days, node_day, side="right"))
        target = sums[rows] / counts[rows]
        target = target + reset_offset * (days[rows] >= data["reset_day"])
        if data.get("third_decon_day") is not None:
            target = target + third_offset * (days[rows] >= float(data["third_decon_day"]))
        led_index = row_led[rows]
        prediction = (
            curves[led_index]
            - base_led_response[led_index, None]
            + led_offsets[led_index, None]
        )
        data_weight = counts[rows] / median_count / noise_sigma**2
        # The production fit defines each LED's reference response from this
        # trajectory and excludes those six reference groups from the
        # likelihood. Keep that same baseline contract in the lookup update.
        data_weight[reference_rows[rows]] = 0.0
        data_cost = 0.5 * np.sum(
            data_weight[:, None] * (prediction - target[:, None]) ** 2,
            axis=0,
        )

        if previous_episode is None or episode != previous_episode:
            if episode == 0:
                start_name = "initial_thickness_log_nm"
            elif episode == 1:
                start_name = "reset_initial_thickness_log_nm"
            else:
                start_name = "third_decon_initial_thickness_log_nm"
            prior_center_log, prior_sigma = _initial_thickness_prior(
                base_archive, start_name
            )
            prior_cost = 0.5 * (
                (np.log(np.maximum(grid, 1.0e-6)) - prior_center_log) / prior_sigma
            ) ** 2
        else:
            dt = max(float(node_day - previous_day), 1.0e-6)
            rate = (grid - previous_thickness) / dt
            valid = rate > 0.0
            prior_cost = np.full(grid.size, np.inf, dtype=float)
            prior_cost[valid] = 0.5 * (
                np.log(rate[valid] / DEFAULT_RATE_MEAN_NM_DAY)
                / DEFAULT_RATE_SIGMA_LOG
            ) ** 2
        seed_day = float(seed_at_nodes[node_index])
        # The H5 history resolves response-curve aliases; it is only a weak
        # branch selector, while the current statistics determine the update.
        seed_scale = max(
            float(branch_selection_sigma_nm),
            0.5 * float(np.median(np.diff(grid))),
        )
        branch_cost = 0.5 * ((grid - seed_day) / seed_scale) ** 2
        total_cost = data_cost + prior_cost + branch_cost
        selected = int(np.argmin(total_cost))
        thickness = float(grid[selected])
        if 0 < selected < grid.size - 1:
            left, center, right = total_cost[selected - 1 : selected + 2]
            curvature = left - 2.0 * center + right
            if np.isfinite(curvature) and curvature > 0.0:
                delta = 0.5 * (left - right) / curvature
                thickness += float(np.clip(delta, -0.5, 0.5)) * float(
                    np.median(np.diff(grid))
                )
        node_history[node_index] = thickness
        rate = (
            max((thickness - previous_thickness) / max(node_day - previous_day, 1.0e-6), 0.0)
            if episode == previous_episode
            else DEFAULT_RATE_MEAN_NM_DAY
        )
        update_records.append(
            {
                "repeat": int(repeat),
                "day": float(node_day),
                "thickness_nm": thickness,
                "rate_nm_day": float(rate),
                "lookup_cost": float(total_cost[selected]),
                "candidate_count": int(grid.size),
                "led_count": int(np.unique(led_index).size),
            }
        )
        previous_day = float(node_day)
        previous_thickness = thickness
        previous_episode = episode

    # Within each physical-deposition episode, project tiny noisy reversals out
    # with a weighted pool-adjacent-violators pass before writing the history.
    for episode in np.unique(np.searchsorted(reset_days, node_days, side="right")):
        selected_nodes = np.flatnonzero(
            np.searchsorted(reset_days, node_days, side="right") == episode
        )
        if selected_nodes.size < 2:
            continue
        node_weights = np.asarray(
            [
                np.sum(counts[np.flatnonzero(row_repeat == repeat)])
                for repeat in repeat_values[selected_nodes]
            ],
            dtype=float,
        )
        node_history[selected_nodes] = _isotonic_increasing(
            node_history[selected_nodes], node_weights
        )

    history = _piecewise_interp(node_days, node_history, days, reset_days)
    node_rates = np.full(node_history.size, DEFAULT_RATE_MEAN_NM_DAY, dtype=float)
    for node_index in range(1, node_history.size):
        same_episode = (
            np.searchsorted(reset_days, node_days[node_index], side="right")
            == np.searchsorted(reset_days, node_days[node_index - 1], side="right")
        )
        if same_episode:
            node_rates[node_index] = max(
                (node_history[node_index] - node_history[node_index - 1])
                / max(node_days[node_index] - node_days[node_index - 1], 1.0e-6),
                0.0,
            )
    rates = _piecewise_interp(node_days, node_rates, days, reset_days)
    return history, rates, node_history, update_records, {
        "led_log_offsets": led_offsets,
        "post_reset_tile_log_offset": reset_offset,
        "post_third_decon_tile_log_offset": third_offset,
    }


def fit_h5_tile_batch(
    data_list,
    curves_list,
    initial_values_list,
    jax,
    jnp,
    *,
    led_log_offset_prior_sigma_log=0.01,
    max_iterations=80,
    initial_damping=1.0e-2,
    gradient_tolerance=1.0e-4,
    rate_knot_days=None,
    thickness_prior_centers_list=None,
    thickness_prior_sigmas_list=None,
):
    """Batched independent spline MAP fits using the H5 lookup forward model.

    The H5 lookup supplies each tile's fixed response curves. Each tile keeps
    its own reset thicknesses, grey offsets, LED offsets, and log-rate spline.
    Supplying a 48-hour knot grid reduces the controls from one per repeat to
    roughly one per two days; reset boundaries are included as explicit knots.
    The batch dimension vectorizes independent residual/Jacobian and block solves.
    """
    from jax.flatten_util import ravel_pytree

    if not data_list or len(data_list) != len(curves_list) or len(data_list) != len(initial_values_list):
        raise ValueError("H5 fit batches require matching non-empty data, curves, and initial values")
    layout = data_list[0]
    shared_fields = (
        "days", "node_days", "led", "repeat", "program", "groups",
        "first_by_led", "repeat_values", "reset_day", "reset_days",
    )
    for tile_data in data_list[1:]:
        for name in shared_fields:
            if not np.array_equal(np.asarray(tile_data[name]), np.asarray(layout[name])):
                raise ValueError(f"Batched H5 fits require identical {name} axes")
    node_days = np.asarray(layout["node_days"], dtype=float)
    control_days = (
        node_days.copy()
        if rate_knot_days is None
        else np.asarray(rate_knot_days, dtype=float).reshape(-1)
    )
    if (
        control_days.size < 2
        or np.any(~np.isfinite(control_days))
        or np.any(np.diff(control_days) <= 0.0)
        or control_days[0] > node_days[0]
        or control_days[-1] < node_days[-1]
    ):
        raise ValueError("Rate spline knots must be finite, increasing, and cover the node-time range")
    ordinary_names = tuple(
        name for name in relative_model.variable_sites(initial_values_list[0])
        if name != "np_log_rate_nodes"
    )
    names = ordinary_names + ("np_log_rate_control_nodes",)
    thickness_names, _offset_names = production.reset_site_names(layout)
    if thickness_prior_centers_list is None:
        thickness_prior_centers_list = [
            {name: float(np.log(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)) for name in thickness_names}
            for _ in data_list
        ]
    if thickness_prior_sigmas_list is None:
        thickness_prior_sigmas_list = [
            {name: float(production.INITIAL_THICKNESS_LOG_SIGMA) for name in thickness_names}
            for _ in data_list
        ]
    if (
        len(thickness_prior_centers_list) != len(data_list)
        or len(thickness_prior_sigmas_list) != len(data_list)
    ):
        raise ValueError("Each tile needs one thickness-prior center/sigma mapping")
    _, unravel = ravel_pytree(
        {
            **{name: jnp.asarray(initial_values_list[0][name]) for name in ordinary_names},
            "np_log_rate_control_nodes": jnp.asarray(
                np.interp(
                    control_days,
                    node_days,
                    np.asarray(initial_values_list[0]["np_log_rate_nodes"], dtype=float).reshape(-1),
                )[None, None, :]
            ),
        }
    )
    flat_initials = []
    flat_scales = []
    rate_references = []
    for tile_index, values in enumerate(initial_values_list):
        if tuple(
            name for name in relative_model.variable_sites(values)
            if name != "np_log_rate_nodes"
        ) != ordinary_names:
            raise ValueError("Batched tiles have different MAP parameter sets")
        parameter_values = {
            name: jnp.asarray(values[name]) for name in ordinary_names
        }
        parameter_values["np_log_rate_control_nodes"] = jnp.asarray(
            np.interp(
                control_days,
                node_days,
                np.asarray(values["np_log_rate_nodes"], dtype=float).reshape(-1),
            )[None, None, :]
        )
        flat_initial, _ = ravel_pytree(parameter_values)
        parameter_scales = production.map_parameter_scale_values(
            values,
            led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
        )
        for name in thickness_names:
            parameter_scales[name] = np.full_like(
                np.asarray(values[name], dtype=float),
                float(thickness_prior_sigmas_list[tile_index][name]),
            )
        parameter_scales["np_log_rate_control_nodes"] = np.full_like(
            np.asarray(parameter_values["np_log_rate_control_nodes"], dtype=float),
            production.TRAJECTORY_RATE_PRIOR_SIGMA_LOG,
        )
        flat_scale, _ = ravel_pytree(
            {name: jnp.asarray(parameter_scales[name]) for name in names}
        )
        flat_initials.append(np.asarray(flat_initial, dtype=float))
        flat_scales.append(np.asarray(flat_scale, dtype=float))
        rate_references.append(
            np.asarray(
                np.full_like(
                    np.asarray(values["np_log_rate_nodes"], dtype=float),
                    np.log(DEFAULT_RATE_MEAN_NM_DAY),
                ),
                dtype=float,
            )
        )
    flat_initials = jnp.asarray(np.stack(flat_initials))
    flat_scales = jnp.asarray(np.stack(flat_scales))
    rate_references = jnp.asarray(np.stack(rate_references))
    thickness_prior_centers = jnp.asarray(
        [
            [float(thickness_prior_centers_list[tile][name]) for name in thickness_names]
            for tile in range(len(data_list))
        ],
        dtype=float,
    )
    thickness_prior_sigmas = jnp.asarray(
        [
            [float(thickness_prior_sigmas_list[tile][name]) for name in thickness_names]
            for tile in range(len(data_list))
        ],
        dtype=float,
    )
    counts = jnp.asarray(np.stack([data["counts"] for data in data_list]))
    sums = jnp.asarray(np.stack([data["sums"] for data in data_list]))
    grid_nm = np.asarray(curves_list[0]["thickness_nm"], dtype=float)
    if any(
        not np.array_equal(np.asarray(entry["thickness_nm"], dtype=float), grid_nm)
        for entry in curves_list
    ):
        raise ValueError("Batched H5 fits require identical thickness grids")
    curve_values = jnp.asarray(
        np.stack([np.asarray(entry["log_response"], dtype=float) for entry in curves_list])
    )
    grid_nm = jnp.asarray(grid_nm)
    offset_names = production.reset_site_names(layout)[1]
    reset_mask = jnp.asarray(
        np.asarray(layout["days"], dtype=float)[None, :]
        >= production.reset_days_for_data(layout)[:, None]
    )

    control_days_jnp = jnp.asarray(control_days)
    node_days_jnp = jnp.asarray(node_days)

    def residual(
        scaled,
        flat_initial,
        flat_scale,
        rate_reference,
        tile_counts,
        tile_sums,
        tile_curves,
        tile_thickness_centers,
        tile_thickness_sigmas,
        grid_nm,
    ):
        values = unravel(flat_initial + flat_scale * scaled)
        control_log_rates = values.pop("np_log_rate_control_nodes").reshape(-1)
        values["np_log_rate_nodes"] = jnp.interp(
            node_days_jnp, control_days_jnp, control_log_rates
        ).reshape(1, 1, -1)
        state = relative_model.relative_state(
            layout,
            values,
            {"grid_nm": grid_nm, "relative_log_response": tile_curves},
            jax,
            jnp,
        )
        reset_offsets = jnp.stack(
            [values[name].reshape(-1) for name in offset_names], axis=0
        )
        observed = tile_sums / tile_counts + jnp.sum(
            reset_offsets[:, :, None] * reset_mask[:, None, :], axis=0
        )
        data_residual = (
            jnp.sqrt(1.0 / jnp.median(tile_counts))
            / production.LOG_NOISE_SIGMA
            * jnp.sqrt(tile_counts)
            * (state["prediction"] - observed)
        ).reshape(-1)
        center_map = {
            name: tile_thickness_centers[index]
            for index, name in enumerate(thickness_names)
        }
        sigma_map = {
            name: tile_thickness_sigmas[index]
            for index, name in enumerate(thickness_names)
        }
        return jnp.concatenate(
            (
                data_residual,
                relative_model._prior_residuals(
                    layout,
                    values,
                    rate_reference,
                    jnp,
                    led_log_offset_prior_sigma_log=led_log_offset_prior_sigma_log,
                    thickness_prior_log_centers=center_map,
                    thickness_prior_log_sigma=sigma_map,
                ),
            )
        )

    one_jacobian = jax.jacfwd(residual, argnums=0)

    def residual_and_jacobian(
        scaled, flat_initial, flat_scale, rate_reference, tile_counts, tile_sums,
        tile_curves, tile_thickness_centers, tile_thickness_sigmas, grid_nm,
    ):
        value = residual(
            scaled, flat_initial, flat_scale, rate_reference,
            tile_counts, tile_sums, tile_curves,
            tile_thickness_centers, tile_thickness_sigmas, grid_nm,
        )
        jacobian = one_jacobian(
            scaled, flat_initial, flat_scale, rate_reference,
            tile_counts, tile_sums, tile_curves,
            tile_thickness_centers, tile_thickness_sigmas, grid_nm,
        )
        return value, jacobian

    residual_batch = jax.jit(
        jax.vmap(residual, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None))
    )
    residual_jacobian_batch = jax.jit(
        jax.vmap(residual_and_jacobian, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None))
    )

    initial_scaled = jnp.zeros_like(flat_initials)
    lambdas = jnp.full((len(data_list),), float(initial_damping), dtype=float)
    done = jnp.zeros((len(data_list),), dtype=bool)
    initial_residuals = residual_batch(
        initial_scaled,
        flat_initials,
        flat_scales,
        rate_references,
        counts,
        sums,
        curve_values,
        thickness_prior_centers,
        thickness_prior_sigmas,
        grid_nm,
    )
    initial_costs = 0.5 * jnp.sum(initial_residuals**2, axis=1)

    def lm_step(carry, _):
        parameters, damping, completed = carry
        residual_values, jacobian = residual_jacobian_batch(
            parameters, flat_initials, flat_scales, rate_references,
            counts, sums, curve_values, thickness_prior_centers,
            thickness_prior_sigmas, grid_nm,
        )
        costs = 0.5 * jnp.sum(residual_values**2, axis=1)
        gradient = jnp.einsum("brp,br->bp", jacobian, residual_values)
        hessian = jnp.einsum("brp,brq->bpq", jacobian, jacobian)
        diagonal = jnp.maximum(jnp.diagonal(hessian, axis1=1, axis2=2), 1.0e-8)
        damped = hessian + jnp.eye(hessian.shape[-1])[None, :, :] * (
            damping[:, None, None] * diagonal[:, None, :]
        )
        delta = jnp.linalg.solve(damped, -gradient[..., None])[..., 0]
        candidate = parameters + delta
        candidate_residual = residual_batch(
            candidate, flat_initials, flat_scales, rate_references,
            counts, sums, curve_values, thickness_prior_centers,
            thickness_prior_sigmas, grid_nm,
        )
        candidate_cost = 0.5 * jnp.sum(candidate_residual**2, axis=1)
        accept = (~completed) & (candidate_cost < costs)
        next_parameters = jnp.where(accept[:, None], candidate, parameters)
        next_damping = jnp.where(
            completed,
            damping,
            jnp.where(
                accept,
                jnp.maximum(damping * 0.3, 1.0e-8),
                jnp.minimum(damping * 10.0, 1.0e8),
            ),
        )
        gradient_norm = jnp.max(jnp.abs(gradient), axis=1)
        step_norm = jnp.linalg.norm(delta, axis=1)
        next_completed = completed | (
            (gradient_norm <= gradient_tolerance)
            | (step_norm <= 1.0e-6 * (1.0 + jnp.linalg.norm(parameters, axis=1)))
        )
        return (next_parameters, next_damping, next_completed), (
            jnp.where(accept, candidate_cost, costs),
            gradient_norm,
            step_norm,
            accept,
            next_completed,
        )

    def run_solver(parameters, iterations):
        return jax.lax.scan(
            lm_step,
            (parameters, lambdas, done),
            xs=None,
            length=iterations,
        )

    solver = jax.jit(run_solver, static_argnums=(1,))
    solver_started = time.perf_counter()
    fitted_scaled, history = solver(initial_scaled, int(max_iterations))
    final_parameters, _final_damping, done = fitted_scaled
    costs, gradients, step_norms, accepted, completion_history = history
    final_parameters.block_until_ready()
    batch_seconds = time.perf_counter() - solver_started

    fitted_values = []
    diagnostics = []
    for index, initial_values in enumerate(initial_values_list):
        variable_values = unravel(
            flat_initials[index] + flat_scales[index] * final_parameters[index]
        )
        control_log_rates = np.asarray(
            variable_values.pop("np_log_rate_control_nodes"), dtype=float
        ).reshape(-1)
        values = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in initial_values.items()
        }
        values.update(
            {name: np.asarray(value, dtype=float) for name, value in variable_values.items()}
        )
        values["np_log_rate_nodes"] = np.interp(
            node_days, control_days, control_log_rates
        )[None, None, :]
        fitted_values.append(values)
        completed_steps = np.flatnonzero(np.asarray(completion_history[:, index]))
        iteration_count = (
            int(completed_steps[0] + 1)
            if completed_steps.size
            else int(max_iterations)
        )
        diagnostics.append(
            {
                "converged": bool(done[index]),
                "message": "batched damped Gauss-Newton",
                "nfev": iteration_count,
                "parameter_count": int(flat_initials.shape[1]),
                "compile_seconds": float(batch_seconds if index == 0 else 0.0),
                "total_seconds": float(batch_seconds / len(data_list)),
                "initial_residual_cost": float(initial_costs[index]),
                "residual_cost": float(costs[-1, index]),
                "trajectory_rate_initialization_nm_day": float(
                    np.exp(np.median(np.asarray(initial_values["np_log_rate_nodes"])))
                ),
                "gradient_inf_norm": float(gradients[-1, index]),
                "last_step_norm": float(step_norms[-1, index]),
            }
        )
    return fitted_values, diagnostics


def _isotonic_increasing(values, weights):
    """Weighted PAVA projection onto nondecreasing histories."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    levels = []
    masses = []
    starts = []
    stops = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        levels.append(float(value))
        masses.append(max(float(weight), 1.0e-12))
        starts.append(index)
        stops.append(index + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            mass = masses[-2] + masses[-1]
            level = (levels[-2] * masses[-2] + levels[-1] * masses[-1]) / mass
            levels[-2:] = [level]
            masses[-2:] = [mass]
            stops[-2:] = [stops[-1]]
            starts.pop()
    output = np.empty_like(values)
    for level, start, stop in zip(levels, starts, stops, strict=True):
        output[start:stop] = level
    return output


def detector_reference_group(data, detector, tile_id):
    return {
        "detector": str(detector).upper(),
        "tile_id": int(tile_id),
        "tile_row": int(data["metadata"][int(tile_id)]["tile_row"]),
        "tile_col": int(data["metadata"][int(tile_id)]["tile_col"]),
    }


def write_lookup_fit_archive(
    source_fit,
    destination,
    statistics_path,
    data,
    history,
    rates,
    node_history,
    node_rates,
    base_h5_sha256,
    lookup_records,
    history_update_method="Eric-core six-LED thickness lookup",
):
    with np.load(source_fit, allow_pickle=False) as source:
        archive = {name: np.asarray(source[name]) for name in source.files}
    archive.update(
        {
            "observation_days": np.asarray(data["days"], dtype=float),
            "observation_led": np.asarray(data["led"], dtype=int),
            "observation_repeat": np.asarray(data["repeat"], dtype=int),
            "observation_program": np.asarray(data["program"], dtype=int),
            "groups": np.asarray(data["groups"], dtype=int),
            "tile_ice_thickness": np.asarray(history, dtype=float)[None, None, :],
            "tile_instantaneous_rate_nm_day": np.asarray(rates, dtype=float)[None, None, :],
            "np_node_thickness": np.asarray(node_history, dtype=float)[None, None, :],
            "node_instantaneous_rate_nm_day": np.asarray(node_rates, dtype=float)[None, None, :],
            "np_log_rate_nodes": np.log(
                np.maximum(np.asarray(node_rates, dtype=float), 1.0e-8)
            )[None, None, :],
            "np_node_multipliers": np.ones((1, 1, len(node_history)), dtype=float),
            "statistics_path": np.asarray(portable_path_string(statistics_path)),
            "statistics_fingerprint": np.asarray(
                relative_fit.statistics_fingerprint(statistics_path)
            ),
            "fit_mode": np.asarray("h5_response_lookup_ice_history_update"),
            "history_update_method": np.asarray(history_update_method),
            "history_update_base_h5_sha256": np.asarray(base_h5_sha256),
            "history_update_lookup_repeat_count": np.asarray(len(lookup_records)),
        }
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **archive)


def write_group_predictions(path, detector, tile_id, data, history, curves, fit_values):
    led_offsets = np.asarray(fit_values["led_log_offsets"], dtype=float).reshape(-1)
    first_by_led = np.asarray(data["first_by_led"], dtype=int)
    baseline_ice = np.asarray(history, dtype=float)[first_by_led]
    baseline = np.asarray(
        [np.interp(baseline_ice[led], curves["thickness_nm"], curves["log_response"][led])
         for led in range(6)],
        dtype=float,
    )
    rows = []
    reset_day = float(data["reset_day"])
    third_day = data.get("third_decon_day")
    for index, group in enumerate(data["groups"]):
        led = int(data["led"][index]) - 1
        predicted = (
            np.interp(
                history[index],
                curves["thickness_nm"],
                curves["log_response"][led],
            )
            - baseline[led]
            + led_offsets[led]
        )
        observed = float(data["sums"][0, index] / data["counts"][0, index])
        if data["days"][index] >= reset_day:
            observed += float(fit_values["post_reset_tile_log_offset"])
        if third_day is not None and data["days"][index] >= float(third_day):
            observed += float(fit_values["post_third_decon_tile_log_offset"])
        rows.append(
            {
                "detector": str(detector).upper(),
                "tile_id": int(tile_id),
                "group": int(group),
                "days": float(data["days"][index]),
                "led": int(data["led"][index]),
                "repeat": int(data["repeat"][index]),
                "observed_log_change": observed,
                "model_log_change": float(predicted),
                "residual_log": observed - float(predicted),
                "ice_thickness_nm": float(history[index]),
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "detector", "tile_id", "group", "days", "led", "repeat",
        "observed_log_change", "model_log_change", "residual_log",
        "ice_thickness_nm",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tile_rate_summary(path, detector, tile_id, data, rates):
    days = np.asarray(data["days"], dtype=float)
    rates = np.asarray(rates, dtype=float)
    reset_day = float(data["reset_day"])
    third_day = data.get("third_decon_day")
    segments = [(float(days.min()), reset_day)]
    if third_day is None:
        segments.append((reset_day, float(days.max())))
    else:
        segments.extend(
            ((reset_day, float(third_day)), (float(third_day), float(days.max())))
        )
    means = []
    for left, right in segments:
        selected = (days >= left) & (days <= right) & np.isfinite(rates)
        means.append(float(np.mean(rates[selected])) if np.any(selected) else 0.0)
    first = float(rates[np.flatnonzero(np.isfinite(rates))[0]])
    last = float(rates[np.flatnonzero(np.isfinite(rates))[-1]])
    metadata = data["metadata"][int(tile_id)]
    fields = (
        "detector", "tile_id", "tile_row", "tile_col",
        "log_rate_offset_p16", "log_rate_offset_median", "log_rate_offset_p84",
        "initial_rate_nm_day_p16", "initial_rate_nm_day_median", "initial_rate_nm_day_p84",
        "final_rate_nm_day_p16", "final_rate_nm_day_median", "final_rate_nm_day_p84",
        "mean_rate_nm_day_p16", "mean_rate_nm_day_median", "mean_rate_nm_day_p84",
    )
    row = {
        "detector": str(detector).upper(),
        "tile_id": int(tile_id),
        "tile_row": int(metadata["tile_row"]),
        "tile_col": int(metadata["tile_col"]),
        "log_rate_offset_p16": 0.0,
        "log_rate_offset_median": 0.0,
        "log_rate_offset_p84": 0.0,
        "initial_rate_nm_day_p16": first,
        "initial_rate_nm_day_median": first,
        "initial_rate_nm_day_p84": first,
        "final_rate_nm_day_p16": last,
        "final_rate_nm_day_median": last,
        "final_rate_nm_day_p84": last,
        "mean_rate_nm_day_p16": float(np.mean(means)),
        "mean_rate_nm_day_median": float(np.mean(means)),
        "mean_rate_nm_day_p84": float(np.mean(means)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def h5_base_epoch_days(model, exposure_times):
    reference = Time(model.t0_utc, format="isot", scale="utc").mjd
    mjd = []
    for value in exposure_times:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1]
        if text.endswith("+00:00"):
            text = text[:-6]
        mjd.append(float(Time(text, format="isot", scale="utc").mjd))
    return (np.asarray(mjd, dtype=float) - float(reference)), np.asarray(mjd, dtype=float)


def initialize_tile_from_h5(model, detector, tile_id, data, values):
    """Use only the base H5 history as a warm start for one tile fit."""
    superpixels = tile_superpixel_indices(model, detector, tile_id)
    # H5 deposition varies smoothly across the TMM tile; sample its four corners
    # and center instead of expanding every superpixel through every epoch.
    sample_positions = np.unique(
        np.asarray(
            [0, superpixels.size // 4, superpixels.size // 2,
             (3 * superpixels.size) // 4, superpixels.size - 1],
            dtype=int,
        )
    )
    selected = superpixels[sample_positions]
    x = (
        np.asarray(model._col[selected], dtype=int) * int(model.superpixel_px)
        + int(model.superpixel_px) // 2
    )
    y = (
        np.asarray(model._row[selected], dtype=int) * int(model.superpixel_px)
        + int(model.superpixel_px) // 2
    )
    _days_from_h5_t0, mjd = h5_base_epoch_days(model, data["exposure_times"])
    reference_mjd = float(np.median(mjd - np.asarray(data["days"], dtype=float)))
    t0_mjd = float(Time(model.t0_utc, format="isot", scale="utc").mjd)
    node_hours = (
        reference_mjd + np.asarray(data["node_days"], dtype=float) - t0_mjd
    ) * 24.0
    sca = int(str(detector).upper()[3:])
    h5_thickness = np.asarray(model.thickness(sca, x, y, node_hours), dtype=float)
    h5_rate = np.asarray(model.rate(sca, x, y, node_hours), dtype=float)
    reset_days = production.reset_days_for_data(data)
    thickness_seed = np.nanmedian(h5_thickness, axis=0)
    rate_seed = np.nanmedian(h5_rate, axis=0)
    fallback_ice = float(production.INITIAL_THICKNESS_LOG_MEDIAN_NM)
    thickness_seed = np.where(
        np.isfinite(thickness_seed) & (thickness_seed > 0.0),
        thickness_seed,
        fallback_ice,
    )
    rate_seed = np.where(
        np.isfinite(rate_seed) & (rate_seed > 0.0), rate_seed, DEFAULT_RATE_MEAN_NM_DAY
    )
    rate_seed = np.clip(rate_seed, 0.1, 100.0)

    thickness_names, _ = production.reset_site_names(data)
    for stage, name in enumerate(thickness_names):
        if stage == 0:
            index = 0
        else:
            candidates = np.flatnonzero(
                np.asarray(data["node_days"], dtype=float) >= reset_days[stage - 1]
            )
            if not candidates.size:
                continue
            index = int(candidates[0])
        values[name][...] = np.log(max(float(thickness_seed[index]), 1.0e-3))
    values["np_log_rate_nodes"][...] = np.log(rate_seed)[None, None, :]
    return values, thickness_seed, rate_seed


def rate_spline_knot_days(data, spacing_days=2.0):
    """Create the shared 48-hour rate-control grid with exact reset knots."""
    spacing_days = float(spacing_days)
    if not np.isfinite(spacing_days) or spacing_days <= 0.0:
        raise ValueError("Rate-spline knot spacing must be positive and finite")
    node_days = np.asarray(data["node_days"], dtype=float)
    left = float(node_days[0])
    right = float(node_days[-1])
    regular = np.arange(left + spacing_days, right, spacing_days, dtype=float)
    resets = production.reset_days_for_data(data)
    resets = resets[(resets > left) & (resets < right)]
    return np.unique(np.r_[left, regular, resets, right])


def lookup_cache_path(output_root, h5_sha256, detector, tile_id):
    return (
        Path(output_root)
        / "production"
        / "caches"
        / DEFAULT_LOOKUP_CACHE
        / h5_sha256[:16]
        / detector.lower()
        / f"tile_{int(tile_id):03d}"
        / "ice_response_lookup.npz"
    )


def load_or_build_h5_lookup(
    output_root,
    optics_fingerprint,
    model,
    core,
    detector,
    tile_id,
    thickness_grid,
    source_cache_roots=(),
):
    cache_path = lookup_cache_path(output_root, optics_fingerprint, detector, tile_id)
    candidates = [cache_path]
    candidates.extend(
        lookup_cache_path(root, optics_fingerprint, detector, tile_id)
        for root in source_cache_roots
        if Path(root).resolve() != Path(output_root).resolve()
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with np.load(candidate, allow_pickle=False) as archive:
            valid = (
                str(np.asarray(archive["h5_optics_fingerprint"]).reshape(-1)[0])
                == optics_fingerprint
                and str(np.asarray(archive["detector"]).reshape(-1)[0]).upper()
                == detector.upper()
                and int(np.asarray(archive["tile_id"]).reshape(-1)[0])
                == int(tile_id)
            )
            if valid:
                thickness = np.asarray(archive["thickness_nm"], dtype=float)
                curves = np.asarray(archive["relative_log_response"], dtype=float)
                if candidate.resolve() != cache_path.resolve():
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    _copy_file_if_needed(candidate, cache_path)
                return thickness, curves, cache_path

    superpixels = tile_superpixel_indices(model, detector, tile_id)
    tile_amplitudes = np.asarray(model._amp[superpixels])
    tile_cos_theta = np.asarray(model._cos_theta[superpixels], dtype=float)
    if (
        np.max(np.abs(tile_amplitudes - tile_amplitudes[:1])) <= 1.0e-7
        and np.max(np.abs(tile_cos_theta - tile_cos_theta[:1])) <= 1.0e-12
    ):
        selected = superpixels[:1]
    else:
        selected = superpixels

    led_nodes = np.asarray(model.led_nodes_nm, dtype=float)
    led_weights = np.asarray(model.led_weight, dtype=float)
    wavelength_nm = led_nodes.reshape(-1)
    amplitudes = np.asarray(model._amp[selected])
    cos_theta = np.asarray(model._cos_theta[selected], dtype=float)
    reflectance = core.reflectance(
        amplitudes,
        np.asarray(model.wavelength_nm, dtype=float),
        np.asarray(model._mean),
        np.asarray(model._modes),
        wavelength_nm,
    )
    _n, _kappa, fresnel = core.optics_at(
        model.wavelength_nm,
        model._n,
        model._kappa,
        model._b,
        wavelength_nm,
    )
    no_ice = np.asarray(core.no_ice_transmission(reflectance, fresnel), dtype=float)
    n_led, node_count = led_nodes.shape
    if n_led != 6 or led_weights.shape != led_nodes.shape:
        raise ValueError("The base H5 does not have the expected six-LED quadrature")
    weighted_no_ice = (
        no_ice.reshape(selected.size, n_led, node_count)
        * led_weights[None, :, :]
    )
    denominator = weighted_no_ice.sum(axis=2)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise ValueError(f"{detector}/tile {tile_id}: invalid H5 LED normalization")

    thickness_grid = np.asarray(thickness_grid, dtype=float).reshape(-1)
    curves = np.empty((n_led, thickness_grid.size), dtype=float)
    chunk_size = 64
    for start in range(0, thickness_grid.size, chunk_size):
        stop = min(start + chunk_size, thickness_grid.size)
        current_grid = thickness_grid[start:stop]
        d_phys = np.broadcast_to(
            current_grid[None, :], (selected.size, current_grid.size)
        )
        response = np.asarray(
            core.relative_response(
                reflectance,
                wavelength_nm,
                d_phys,
                cos_theta,
                model.wavelength_nm,
                model._n,
                model._kappa,
                model._b,
                model._n_ref,
                model._snell,
            ),
            dtype=float,
        ).reshape(selected.size, n_led, node_count, current_grid.size)
        weights = weighted_no_ice[:, :, :, None]
        per_cell = np.sum(response * weights, axis=2) / denominator[:, :, None]
        if np.any(per_cell <= 0.0) or not np.isfinite(per_cell).all():
            raise ValueError(f"{detector}/tile {tile_id}: invalid H5 response lookup")
        curves[:, start:stop] = np.mean(np.log(per_cell), axis=0)
    if np.max(np.abs(curves[:, thickness_grid == 0.0])) > 1.0e-8:
        raise ValueError(f"{detector}/tile {tile_id}: lookup is not unity at zero ice")

    if tile_id % 64 == 0 and hasattr(model, "led_band_response"):
        valid = np.asarray(model.valid(model._t_epoch), dtype=bool).reshape(-1)
        valid_hours = np.asarray(model._t_epoch, dtype=float)[valid]
        if valid_hours.size:
            sample_hour = np.asarray([valid_hours[-1]], dtype=float)
            sca = int(str(detector).upper()[3:])
            sample_rows = np.asarray(model._row[selected], dtype=int)
            sample_cols = np.asarray(model._col[selected], dtype=int)
            x = sample_cols * model.superpixel_px + model.superpixel_px // 2
            y = sample_rows * model.superpixel_px + model.superpixel_px // 2
            sample_thickness = float(
                np.median(model.thickness(sca, x, y, sample_hour)[:, 0])
            )
            direct = np.asarray(
                model.led_band_response(sca, x, y, sample_hour, gray=False),
                dtype=float,
            )[:, 0, :]
            direct_mean = np.exp(np.mean(np.log(direct), axis=0))
            table_value = np.exp(
                [
                    np.interp(sample_thickness, thickness_grid, curves[led])
                    for led in range(n_led)
                ]
            )
            np.testing.assert_allclose(
                table_value,
                direct_mean,
                rtol=1.0e-4,
                atol=1.0e-5,
                err_msg=f"{detector}/tile {tile_id}: H5 LED lookup differs from Eric reader",
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        h5_optics_fingerprint=np.asarray(optics_fingerprint),
        detector=np.asarray(detector.upper()),
        tile_id=np.asarray(int(tile_id)),
        thickness_nm=thickness_grid,
        relative_log_response=curves,
        lookup_superpixel_count=np.asarray(int(selected.size)),
    )
    return thickness_grid, curves, cache_path


def update_fit_archive(
    base_fit_path,
    output_fit_path,
    statistics_path,
    data,
    history,
    rates,
    node_history,
    node_rates,
    h5_sha256,
):
    with np.load(base_fit_path, allow_pickle=False) as source:
        archive = {name: np.asarray(source[name]) for name in source.files}
    archive.update(
        {
            "map_only": np.asarray(1),
            "observation_days": np.asarray(data["days"], dtype=float),
            "observation_led": np.asarray(data["led"], dtype=int),
            "observation_repeat": np.asarray(data["repeat"], dtype=int),
            "observation_program": np.asarray(data["program"], dtype=int),
            "groups": np.asarray(data["groups"], dtype=int),
            "tile_ice_thickness": np.asarray(history, dtype=float)[None, None, :],
            "tile_instantaneous_rate_nm_day": np.asarray(rates, dtype=float)[None, None, :],
            "np_node_thickness": np.asarray(node_history, dtype=float)[None, None, :],
            "node_instantaneous_rate_nm_day": np.asarray(node_rates, dtype=float)[None, None, :],
            "np_log_rate_nodes": np.log(
                np.maximum(np.asarray(node_rates, dtype=float), 1.0e-8)
            )[None, None, :],
            "np_node_multipliers": np.ones((1, 1, len(node_history)), dtype=float),
            "statistics_path": np.asarray(portable_path_string(statistics_path)),
            "statistics_fingerprint": np.asarray(
                relative_fit.statistics_fingerprint(statistics_path)
            ),
            "fit_mode": np.asarray("h5_response_lookup_ice_history_update"),
            "history_update_base_h5_sha256": np.asarray(h5_sha256),
            "history_update_method": np.asarray("Eric-core six-LED thickness lookup"),
        }
    )
    output_fit_path = Path(output_fit_path)
    output_fit_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_fit_path, **archive)


def update_bundle_trajectories(source_path, output_path, trajectories, metadata_update):
    bundle = bundle_io.load_bundle(source_path)
    arrays = {name: value for name, value in bundle.items() if name != "metadata"}
    detectors = np.asarray(bundle["detectors"]).astype(str)
    tile_ids = np.asarray(bundle["tile_ids"], dtype=int)
    thickness = np.empty(
        (tile_ids.size, len(trajectories[DETECTORS[0]][0]["days"])), dtype=np.float32
    )
    rates = np.empty_like(thickness)
    reset_offsets = np.empty(tile_ids.size, dtype=np.float64)
    third_offsets = np.empty(tile_ids.size, dtype=np.float64)
    third_offsets_present = True
    for index, (detector, tile_id) in enumerate(zip(detectors, tile_ids, strict=True)):
        record = trajectories[detector][int(tile_id)]
        thickness[index] = record["history"]
        rates[index] = record["rates"]
        reset_offsets[index] = record["post_reset_tile_log_offset"]
        if record["post_third_decon_tile_log_offset"] is None:
            third_offsets_present = False
            third_offsets[index] = 0.0
        else:
            third_offsets[index] = record["post_third_decon_tile_log_offset"]
    reference = trajectories[DETECTORS[0]][0]
    arrays["observation_days"] = np.asarray(reference["days"], dtype=np.float64)
    arrays["observation_start_time"] = np.asarray(reference["exposure_times"]).astype(str)
    arrays["observation_led"] = np.asarray(reference["led"], dtype=np.int16)
    arrays["observation_repeat"] = np.asarray(reference["repeat"], dtype=np.int32)
    arrays["observation_program"] = np.asarray(reference["program"], dtype=np.int32)
    arrays["groups"] = np.asarray(reference["groups"], dtype=np.int32)
    arrays["tile_ice_thickness"] = thickness
    arrays["tile_instantaneous_rate_nm_day"] = rates
    arrays["post_reset_tile_log_offset"] = reset_offsets
    arrays["post_reset_tile_throughput_factor"] = np.exp(reset_offsets).astype(
        np.float32
    )
    if third_offsets_present:
        arrays["post_third_decon_tile_log_offset"] = third_offsets
        arrays["post_third_decon_tile_throughput_factor"] = np.exp(
            third_offsets
        ).astype(np.float32)
    metadata = dict(bundle["metadata"])
    metadata["ice_history_update"] = metadata_update
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def _write_dataset(group, name, values, dtype=None):
    if name in group:
        del group[name]
    array = np.asarray(values, dtype=dtype)
    kwargs = {}
    if array.ndim and array.size > 32:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    group.create_dataset(name, data=array, **kwargs)


def make_history_products(base_model, base_h5, output_root, modulated_bundle_path, bundle):
    """Write full-history sidecar and the unchanged-reader rank-one H5 projection."""
    detectors = np.asarray(bundle["detectors"]).astype(str)
    tile_ids = np.asarray(bundle["tile_ids"], dtype=int)
    thickness = np.asarray(bundle["tile_ice_thickness"], dtype=float)
    days_from_stats = np.asarray(bundle["observation_days"], dtype=float)
    timestamps = np.asarray(bundle["observation_start_time"]).astype(str)
    if thickness.shape != (tile_ids.size, days_from_stats.size):
        raise ValueError("Updated bundle history does not match its tile/time axes")
    if timestamps.shape != days_from_stats.shape:
        raise ValueError("Updated bundle timestamps do not match its observation days")

    parsed_mjd = []
    for stamp in timestamps:
        text = str(stamp).strip()
        if text.endswith("Z"):
            text = text[:-1]
        if text.endswith("+00:00"):
            text = text[:-6]
        parsed_mjd.append(float(Time(text, format="isot", scale="utc").mjd))
    parsed_mjd = np.asarray(parsed_mjd, dtype=float)
    new_reference_mjd = float(parsed_mjd[0])
    bundle_days = np.asarray(bundle["observation_days"], dtype=float)
    if bundle_days.shape != parsed_mjd.shape:
        raise ValueError("Bundle and update statistics have different observation axes")
    statistics_reference_mjd = float(np.median(parsed_mjd - bundle_days))
    history_days = parsed_mjd - new_reference_mjd
    history_hours = history_days * 24.0
    reset_days = [float(np.asarray(bundle["ice_reset_day"]).reshape(-1)[0])]
    if "third_decon_day" in bundle:
        reset_days.append(float(np.asarray(bundle["third_decon_day"]).reshape(-1)[0]))
    gray_hours = (
        statistics_reference_mjd
        + np.asarray(reset_days, dtype=float)
        - new_reference_mjd
    ) * 24.0
    time_shift_hours = (
        float(Time(base_model.t0_utc, format="isot", scale="utc").mjd)
        - new_reference_mjd
    ) * 24.0
    episodes = np.searchsorted(gray_hours, history_hours, side="right").astype(np.int8)
    factorization = deposition_factorizations_on_blocks(
        history_days,
        episodes,
        thickness,
        spacing_days=2.0,
        multi_rank=16,
    )

    history_root = (
        Path(output_root)
        / "production/wavelength_response_bundle/intermediates/h5"
    )
    history_path = history_root / "joint_ice_history_reference.h5"
    history_root.mkdir(parents=True, exist_ok=True)
    base_sha = sha256_file(base_h5)
    bundle_sha = sha256_file(modulated_bundle_path)
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(history_path, "w") as h5:
        h5.attrs["schema"] = "rcs_joint_ice_history_reference/1"
        h5.attrs["product_type"] = "full per-tile ice history from H5 response lookup"
        h5.attrs["v3_deposition_compatible_parameters_included"] = True
        h5.attrs["rate_control_spacing_days"] = 2.0
        h5.attrs["ice_reference_utc"] = str(timestamps[0])
        h5.attrs["reset_days_from_reference"] = gray_hours / 24.0
        h5.attrs["source_bundle"] = str(Path(modulated_bundle_path).resolve())
        h5.attrs["source_bundle_sha256"] = bundle_sha
        h5.attrs["base_h5"] = str(Path(base_h5).resolve())
        h5.attrs["base_h5_sha256"] = base_sha

        superpixels = h5.create_group("superpixels")
        superpixels.create_dataset(
            "detector", data=np.asarray(detectors, dtype=string_type)
        )
        superpixels.create_dataset("tile_id", data=tile_ids.astype(np.int32))
        superpixels.attrs["grid_bins_per_side"] = 16
        superpixels.attrs["tile_size_pixels"] = TILE_SIZE_PIXELS
        superpixels.attrs["tile_order"] = "SCA-major, row-major tile ID"

        observation = h5.create_group("bundle_observation_axis")
        _write_dataset(observation, "time_days_from_reference", history_days)
        _write_dataset(observation, "time_hours_from_reference", history_hours)
        observation.create_dataset("timestamp_utc", data=np.asarray(timestamps, dtype=string_type))
        _write_dataset(observation, "episode", episodes.astype(np.int8))
        _write_dataset(observation, "tile_thickness_nm", thickness.astype(np.float32))
        _write_dataset(
            observation,
            "tile_rate_nm_per_day",
            np.asarray(bundle["tile_instantaneous_rate_nm_day"], dtype=np.float32),
        )

        rank1 = h5.create_group("v3_rank1_deposition_candidate")
        candidate = factorization["rank1"]
        _write_dataset(rank1, "time_hours", factorization["time_hours"])
        _write_dataset(rank1, "episode", factorization["episodes"].astype(np.int8))
        _write_dataset(rank1, "blocks", factorization["blocks"])
        _write_dataset(rank1, "multipliers", candidate["multipliers"])
        _write_dataset(rank1, "tau_h", np.empty(0, dtype=float))
        _write_dataset(rank1, "offset_episodes", np.empty(0, dtype=np.int16))
        _write_dataset(rank1, "log_a", candidate["log_a"])
        _write_dataset(rank1, "d_off", candidate["d_off_nm"])
        _write_dataset(rank1, "rate_nm_per_day", candidate["rate_nm_per_day"])
        rank1.attrs["current_v3_deposition_group_fields"] = True
        rank1.attrs["fit_metrics_json"] = json.dumps(candidate["metrics"], sort_keys=True)
        rank1.attrs["warning"] = (
            "Rank-one approximation to the full tile/epoch histories; the complete "
            "histories are in bundle_observation_axis/tile_thickness_nm."
        )
        h5.attrs["rank1_projection_per_tile_max_error_nm_p50_p95_max"] = (
            candidate["metrics"]["per_tile_max_thickness_error_nm_p50_p95_max"]
        )
    return history_path, factorization, parsed_mjd, episodes, gray_hours, time_shift_hours


def update_reader_h5(base_h5, output_h5, base_model, bundle, factorization, timestamps, gray_hours, time_shift_hours):
    """Copy the base model and replace only its v3-compatible rank-one history."""
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_h5.with_name(output_h5.name + ".update.tmp")
    shutil.copy2(base_h5, temporary)

    detectors = np.asarray(bundle["detectors"]).astype(str)
    tile_ids = np.asarray(bundle["tile_ids"], dtype=int)
    tile_index = {
        (detector, int(tile_id)): index
        for index, (detector, tile_id) in enumerate(zip(detectors, tile_ids, strict=True))
    }
    factor = int(TILE_SIZE_PIXELS // base_model.superpixel_px)
    fine_sca = np.asarray(base_model._sca, dtype=int)
    fine_row = np.asarray(base_model._row, dtype=int)
    fine_col = np.asarray(base_model._col, dtype=int)
    fine_tile_id = (fine_row // factor) * (4096 // TILE_SIZE_PIXELS) + fine_col // factor
    fine_tile_index = np.asarray(
        [tile_index[(f"WFI{sca:02d}", int(tile))] for sca, tile in zip(fine_sca, fine_tile_id, strict=True)],
        dtype=int,
    )
    rank1 = factorization["rank1"]
    with h5py.File(temporary, "r+") as h5:
        old_deposition = h5["deposition"]
        old_attrs = dict(old_deposition.attrs)
        old_covariance = (
            np.asarray(old_deposition["rate_param_cov"])
            if "rate_param_cov" in old_deposition
            else None
        )
        del h5["deposition"]
        deposition = h5.create_group("deposition")
        for name, value in old_attrs.items():
            deposition.attrs[name] = value
        deposition.attrs["t0_utc"] = str(timestamps[0])
        deposition.attrs["current_joint_history_reference_utc"] = str(timestamps[0])
        deposition.attrs["v3_parameterization"] = (
            "single spatial log_a map times shared 2-day rate-block multipliers, "
            "with per-episode d_off; full independent histories are stored in the "
            "joint_ice_history_reference.h5 sidecar"
        )
        deposition.attrs["ice_history_update_method"] = "H5 six-LED lookup"
        _write_dataset(deposition, "time_hours", factorization["time_hours"])
        _write_dataset(deposition, "episode", factorization["episodes"].astype(np.int8))
        deposition.create_dataset(
            "timestamp_utc", data=np.asarray(timestamps, dtype="S32")
        )
        _write_dataset(deposition, "blocks", factorization["blocks"])
        _write_dataset(deposition, "multipliers", rank1["multipliers"])
        _write_dataset(deposition, "tau_h", np.empty(0, dtype=float))
        _write_dataset(deposition, "offset_episodes", np.empty(0, dtype=np.int16))
        _write_dataset(deposition, "log_a", rank1["log_a"][fine_tile_index].astype(np.float32))
        _write_dataset(deposition, "d_off", rank1["d_off_nm"][fine_tile_index].astype(np.float32))
        rates = rank1["rate_nm_per_day"][fine_tile_index].astype(np.float32)
        _write_dataset(deposition, "rate_nm_per_day", rates)
        if old_covariance is not None:
            _write_dataset(deposition, "rate_param_cov", np.zeros_like(old_covariance))
            deposition.attrs["rate_param_covariance_note"] = (
                "Not refit by lookup update; stored as zeros."
            )
        deposition.attrs["rank1_history_fit_metrics_json"] = json.dumps(
            rank1["metrics"], sort_keys=True
        )

        gray = h5["instrument/gray_period_hours"]
        gray[...] = np.asarray(gray_hours, dtype=float)
        if "step_boundary_hours" in h5["instrument"]:
            h5["instrument/step_boundary_hours"][...] = (
                np.asarray(h5["instrument/step_boundary_hours"], dtype=float)
                + float(time_shift_hours)
            )
        if "time" in h5:
            time_group = h5["time"]
            episodes = np.asarray(factorization["episodes"], dtype=int)
            utc = np.asarray(timestamps, dtype="S32")
            for prefix, index_fn in (("first", np.min), ("last", np.max)):
                if f"episode_{prefix}_hours" in time_group:
                    values = np.asarray([
                        index_fn(factorization["time_hours"][episodes == episode])
                        for episode in range(gray_hours.size + 1)
                    ], dtype=float)
                    time_group[f"episode_{prefix}_hours"][...] = values
                if f"episode_{prefix}_utc" in time_group:
                    values = []
                    for episode in range(gray_hours.size + 1):
                        selected = np.flatnonzero(episodes == episode)
                        values.append(utc[selected[0] if prefix == "first" else selected[-1]])
                    time_group[f"episode_{prefix}_utc"][...] = np.asarray(values, dtype="S32")
        h5.attrs["ice_history_update_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["ice_history_update_rank1_projection"] = True
    os.replace(temporary, output_h5)
    return output_h5


def write_history_reference(
    output_path,
    source_bundle,
    base_h5,
    base_h5_sha256,
    bundle,
    histories,
    rates,
    reference_utc,
    times_days,
    timestamps,
    episodes,
    reset_days_from_reference,
    factorization,
    lookup_summary,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_bundle = Path(source_bundle)
    bundle_sha = sha256_file(source_bundle)
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema"] = "rcs_joint_ice_history_reference/1"
        h5.attrs["product_type"] = "full independent per-tile H5-LUT ice histories"
        h5.attrs["v3_deposition_compatible_parameters_included"] = True
        h5.attrs["rate_control_spacing_days"] = 2.0
        h5.attrs["ice_reference_utc"] = str(reference_utc)
        h5.attrs["reset_days_from_reference"] = np.asarray(
            reset_days_from_reference, dtype=float
        )
        h5.attrs["source_bundle"] = str(source_bundle.resolve())
        h5.attrs["source_bundle_sha256"] = bundle_sha
        h5.attrs["base_h5"] = str(Path(base_h5).resolve())
        h5.attrs["base_h5_sha256"] = str(base_h5_sha256)
        h5.attrs["history_method"] = "H5 optical LUT with fixed saved LED offsets"

        superpixels = h5.create_group("superpixels")
        superpixels.create_dataset(
            "detector", data=np.asarray(bundle["detectors"]).astype(str).astype(string_type)
        )
        for name in (
            "tile_ids", "sca_numbers", "tile_rows", "tile_cols",
            "row_start", "row_stop_exclusive", "col_start", "col_stop_exclusive",
            "tile_center_x_mm", "tile_center_y_mm",
        ):
            if name in bundle:
                _write_dataset(superpixels, name, bundle[name])
        superpixels.attrs["grid_bins_per_side"] = 16
        superpixels.attrs["tile_size_pixels"] = TILE_SIZE_PIXELS
        superpixels.attrs["tile_order"] = "SCA-major, row-major tile ID"

        observation = h5.create_group("bundle_observation_axis")
        _write_dataset(observation, "time_days_from_reference", times_days)
        _write_dataset(observation, "time_hours_from_reference", times_days * 24.0)
        observation.create_dataset("timestamp_utc", data=np.asarray(timestamps, dtype=string_type))
        _write_dataset(observation, "episode", factorization["episodes"].astype(np.int8))
        _write_dataset(observation, "tile_thickness_nm", histories.astype(np.float32))
        _write_dataset(observation, "tile_rate_nm_per_day", rates.astype(np.float32))

        rank1 = h5.create_group("v3_rank1_deposition_candidate")
        candidate = factorization["rank1"]
        _write_dataset(rank1, "time_hours", factorization["time_hours"])
        _write_dataset(rank1, "episode", factorization["episodes"].astype(np.int8))
        _write_dataset(rank1, "blocks", factorization["blocks"])
        _write_dataset(rank1, "multipliers", candidate["multipliers"])
        _write_dataset(rank1, "tau_h", np.empty(0, dtype=float))
        _write_dataset(rank1, "offset_episodes", np.empty(0, dtype=np.int16))
        _write_dataset(rank1, "log_a", candidate["log_a"])
        _write_dataset(rank1, "d_off", candidate["d_off_nm"])
        _write_dataset(rank1, "rate_nm_per_day", candidate["rate_nm_per_day"])
        rank1.attrs["current_v3_deposition_group_fields"] = True
        rank1.attrs["fit_metrics_json"] = json.dumps(candidate["metrics"], sort_keys=True)
        rank1.attrs["warning"] = (
            "Rank-one H5 projection; full independent histories are retained in "
            "bundle_observation_axis/tile_thickness_nm."
        )

        lookup = h5.create_group("h5_lookup_update")
        lookup.attrs["method"] = "six-LED likelihood lookup with rate/reset continuity"
        lookup.attrs["rate_prior_mean_nm_day"] = DEFAULT_RATE_MEAN_NM_DAY
        lookup.attrs["rate_prior_sigma_log"] = DEFAULT_RATE_SIGMA_LOG
        lookup.attrs["tile_update_summary_json"] = json.dumps(lookup_summary, sort_keys=True)


def _write_updated_manifest(output_root, base_h5, statistics_dir, output_h5, history_file, report):
    output_root = Path(output_root)
    manifest_path = output_root / "pipeline_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    else:
        manifest = {}
    manifest.setdefault("pipeline", "wfi_relative_response")
    manifest["ice_history_update"] = {
        "status": "complete",
        "method": "h5_six_led_lookup",
        "base_h5": str(Path(base_h5).resolve()),
        "statistics_dir": str(Path(statistics_dir).resolve()),
        "output_h5": str(Path(output_h5).resolve()),
        "full_history_reference": str(Path(history_file).resolve()),
        "report": report,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _copy_file_if_needed(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return destination
    try:
        os.link(source, destination)
    except FileExistsError:
        return destination
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _write_validation(base_h5, output_h5, base_h5_sha256, history_path, report):
    with reader_loader.load_eric_v3_reader(reader_loader.DEFAULT_SOURCE_ZIP) as reader:
        model = reader.IceResponseModel(output_h5)
        wavelength = np.asarray([620.0, 1038.0, 1670.0, 2200.0])
        valid_hours = np.asarray(model._t_epoch, dtype=float)
        valid_mask = np.asarray(model.valid(valid_hours), dtype=bool).reshape(-1)
        if not np.any(valid_mask):
            raise ValueError("Updated H5 has no model-valid deposition epochs")
        modelled_hours = valid_hours[valid_mask]
        sample_hours = np.asarray([modelled_hours[0], modelled_hours[-1]], dtype=float)
        finite_by_detector = {}
        for detector in DETECTORS:
            sca = int(detector[3:])
            position = np.flatnonzero(model._sca == sca)[0]
            x = int(model._col[position] * model.superpixel_px + model.superpixel_px // 2)
            y = int(model._row[position] * model.superpixel_px + model.superpixel_px // 2)
            values = np.asarray(model.response(sca, [x], [y], sample_hours, wavelength, gray=True))
            if not np.isfinite(values).all():
                raise ValueError(f"{detector}: updated H5 has nonfinite response samples")
            finite_by_detector[detector] = True
        report["eric_reader_load"] = True
        report["finite_reader_responses_by_detector"] = finite_by_detector
    report["base_h5"] = str(Path(base_h5).resolve())
    report["base_h5_sha256"] = base_h5_sha256
    report["output_h5"] = str(Path(output_h5).resolve())
    report["history_reference"] = str(Path(history_path).resolve())
    validation_path = Path(output_h5).with_suffix(".validation.json")
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return validation_path


def _write_lookup_summary_csv(path, records):
    fields = (
        "detector", "tile_id", "repeat", "day", "thickness_nm",
        "rate_nm_day", "lookup_cost", "candidate_count", "led_count",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _tile_lookup_records(records, detector, tile_id):
    for record in records:
        yield {"detector": detector, "tile_id": int(tile_id), **record}


def _run_focal_plane_and_ice_trend_plots(
    statistics_dir,
    plot_root,
    detector_fit_roots,
    reset_day,
    third_decon_day,
):
    """Generate the full-SCA mosaics and combined ice-history diagnostics."""
    repository_root = Path(__file__).resolve().parents[1]
    archived_plot_root = (
        repository_root / "archive/exploratory/legacy_root_scripts"
    )
    focal_plane_script = archived_plot_root / "plot_wfi_full_focal_plane_rates.py"
    trajectories_script = archived_plot_root / "plot_wfi_all_detector_trajectories.py"
    for script in (focal_plane_script, trajectories_script):
        if not script.is_file():
            raise FileNotFoundError(f"Required production diagnostic plotter is missing: {script}")

    plot_root = Path(plot_root)
    focal_plane_root = plot_root / "focal_plane"
    trajectories_root = plot_root / "trajectories"
    recent_trajectories_root = plot_root / "trajectories_last_7d"
    for directory in (focal_plane_root, trajectories_root, recent_trajectories_root):
        directory.mkdir(parents=True, exist_ok=True)

    focal_plane_output = focal_plane_root / "wfi_full_focal_plane_rate_maps.png"
    aom_output = focal_plane_root / "wfi_full_focal_plane_aom_rate_jump_percent.png"
    full_trajectory_output = (
        trajectories_root / "wfi_all_detector_ice_rate_trajectories.png"
    )
    recent_trajectory_output = (
        recent_trajectories_root
        / "wfi_all_detector_ice_rate_trajectories_last_7d.png"
    )
    plot_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    python_path = [
        str(repository_root / "wfi_pipeline"),
        str(repository_root),
    ]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    with tempfile.TemporaryDirectory(
        prefix=".ice-history-plot-input-", dir=plot_root
    ) as temporary_root:
        detector_root = Path(temporary_root)
        for detector in DETECTORS:
            source = Path(detector_fit_roots[detector]).resolve()
            if not source.is_dir():
                raise FileNotFoundError(source)
            (detector_root / detector.lower()).symlink_to(source, target_is_directory=True)

        common = [
            "--root", str(Path(statistics_dir).resolve()),
            "--tile-size", str(TILE_SIZE_PIXELS),
        ]
        focal_command = [
            sys.executable,
            str(focal_plane_script),
            *common,
            "--fit-root", str(detector_root),
            "--tile-statistics", str(Path(statistics_dir).resolve()),
            "--output", str(focal_plane_output),
            "--aom-rate-jump-output", str(aom_output),
        ]
        trajectory_base = [
            sys.executable,
            str(trajectories_script),
            *common,
            "--detector-output-root", str(detector_root),
            "--reset-day", str(float(reset_day)),
        ]
        if third_decon_day is not None:
            trajectory_base.extend(("--break-day", str(float(third_decon_day))))

        commands = (
            focal_command,
            [*trajectory_base, "--output", str(full_trajectory_output)],
            [
                *trajectory_base,
                "--output", str(recent_trajectory_output),
                "--last-days", "7",
            ],
        )
        for command in commands:
            subprocess.run(
                command,
                check=True,
                cwd=repository_root,
                env=environment,
            )

    expected_outputs = [
        focal_plane_output,
        focal_plane_output.with_name("wfi_full_focal_plane_rate_maps_tile_contrast.png"),
        focal_plane_output.with_name("wfi_full_focal_plane_initial_thicknesses.png"),
        focal_plane_output.with_name("wfi_full_focal_plane_current_ice_thickness.png"),
        focal_plane_output.with_name("wfi_full_focal_plane_rate_maps_pre_post_reset.png"),
        focal_plane_output.with_name("wfi_full_focal_plane_rate_maps_pre_post_reset_tile_contrast.png"),
        focal_plane_output.with_name("wfi_full_focal_plane_trend_rates_pre_post.png"),
        focal_plane_output.with_name("wfi_full_focal_plane_trend_rates_pre_post_tile_contrast.png"),
        focal_plane_output.with_name("wfi_full_focal_plane_mean_accumulation_rate_last_48h.png"),
        focal_plane_output.with_name("wfi_full_focal_plane_mean_accumulation_rate_last_48h_tile_contrast.png"),
        focal_plane_output.with_name("wfi_full_focal_plane_rms_residual.png"),
        aom_output,
        full_trajectory_output,
        full_trajectory_output.with_name("wfi_all_detector_ice_rate_trajectories_quantiles.csv"),
        full_trajectory_output.with_name("wfi_all_detector_detrended_ice_thickness.png"),
        full_trajectory_output.with_name("wfi_all_detector_detrended_ice_thickness_fraction.png"),
        full_trajectory_output.with_name("wfi_all_tile_detrended_ice_thickness.png"),
        full_trajectory_output.with_name("wfi_all_tile_detrended_ice_thickness_fraction.png"),
        recent_trajectory_output,
        recent_trajectory_output.with_name(
            "wfi_all_detector_ice_rate_trajectories_last_7d_quantiles.csv"
        ),
        recent_trajectory_output.with_name("wfi_all_detector_detrended_ice_thickness.png"),
        recent_trajectory_output.with_name(
            "wfi_all_detector_detrended_ice_thickness_fraction.png"
        ),
        recent_trajectory_output.with_name("wfi_all_tile_detrended_ice_thickness.png"),
        recent_trajectory_output.with_name(
            "wfi_all_tile_detrended_ice_thickness_fraction.png"
        ),
    ]
    histogram_outputs = tuple(
        focal_plane_root.glob("wfi_full_focal_plane_rate_histograms_*.png")
    )
    if not histogram_outputs:
        raise RuntimeError("Focal-plane plotting did not produce a rate-histogram mosaic")
    expected_outputs.extend(histogram_outputs)
    missing = [path for path in expected_outputs if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Focal-plane/ice-trend plot generation omitted expected outputs: "
            + ", ".join(str(path) for path in missing)
        )
    return [str(path.resolve()) for path in expected_outputs]


def _legacy_main_streaming(argv=None):
    args = parse_args(argv)
    base_h5 = args.base_h5
    statistics_dir = args.statistics_dir
    output_root = args.output_path
    bundle_root = find_bundle_root(base_h5)
    base_run_root = find_base_run_root(base_h5, bundle_root, args.base_run_root)
    base_h5_sha256 = sha256_file(base_h5)
    base_bundle_path = bundle_root / "wfi_all_2nm_rank16.npz"
    modulated_bundle_path = bundle_root / "wfi_all_2nm_rank16_modulated.npz"
    if not base_bundle_path.is_file() or not modulated_bundle_path.is_file():
        raise FileNotFoundError(f"Missing stable wavelength bundles in {bundle_root}")
    base_bundle = bundle_io.load_bundle(base_bundle_path)
    modulated_bundle = bundle_io.load_bundle(modulated_bundle_path)
    thickness_grid = np.asarray(modulated_bundle["ice_thickness_nm"], dtype=float)
    if thickness_grid.ndim != 1 or thickness_grid.size < 2 or np.any(np.diff(thickness_grid) <= 0):
        raise ValueError("Base modulated bundle has an invalid thickness grid")

    output_root.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_root, RELATIVE_RESPONSE)
    for path in (paths["logs"], paths["production"], paths["bundle_root"]):
        path.mkdir(parents=True, exist_ok=True)
    _copy_spectral_caches(base_run_root, output_root)

    detector_data = {}
    detector_tile_paths = {}
    for detector in args.detectors:
        stats_path = corrected_statistics_path(
            statistics_dir, detector, TILE_SIZE_PIXELS
        )
        if not stats_path.is_file():
            raise FileNotFoundError(stats_path)
        relative_root = paths["production"] / detector.lower() / "relative_response"
        detector_tile_paths[detector] = relative_runner.tile_statistics_paths(
            detector,
            stats_path,
            relative_root / "tile_statistics",
            TILE_SIZE_PIXELS,
        )
        complete_tile_ids(
            detector_tile_paths[detector], label=f"{detector} statistics tile IDs"
        )
        detector_data[detector] = {
            tile_id: production.load_tile_statistics(
                stats_file, third_decon_day=THIRD_DECON_DAY
            )
            for tile_id, stats_file in detector_tile_paths[detector].items()
        }

    reference_data = detector_data["WFI01"][0]
    rate_control_days = rate_spline_knot_days(
        reference_data, spacing_days=args.rate_knot_spacing_hours / 24.0
    )
    for detector in args.detectors:
        for tile_id, data in detector_data[detector].items():
            for name in (
                "days", "led", "repeat", "program", "groups", "exposure_times"
            ):
                if not np.array_equal(reference_data[name], data[name]):
                    raise ValueError(
                        f"{detector}/tile {tile_id}: statistics observation layout differs in {name}"
                    )

    trajectories = {detector: {} for detector in DETECTORS}
    output_fit_roots = {}
    lookup_records_path = (
        paths["bundle_root"] / "ice_history_lookup_nodes.csv"
    )
    total_lookup_nodes = 0
    total_history_nodes = 0
    reused_base_fit_tiles = 0
    h5_lookup_tiles = 0
    boundary_hits = 0

    with reader_loader.load_eric_v3_reader(reader_loader.DEFAULT_SOURCE_ZIP) as reader:
        model = reader.IceResponseModel(base_h5)
        optics_fingerprint = h5_optical_fingerprint(model)
        base_output = output_root / "production/wavelength_response_bundle/final"
        base_output.mkdir(parents=True, exist_ok=True)
        with lookup_records_path.open("w", newline="") as lookup_csv:
            lookup_fields = (
                "detector", "tile_id", "repeat", "day", "thickness_nm",
                "rate_nm_day", "lookup_cost", "candidate_count", "led_count",
            )
            lookup_writer = csv.DictWriter(lookup_csv, fieldnames=lookup_fields)
            lookup_writer.writeheader()
            for detector in args.detectors:
                fit_root = paths["production"] / detector.lower() / "relative_response" / "fits"
                output_fit_roots[detector] = fit_root
                for tile_id, stats_path in detector_tile_paths[detector].items():
                    base_fit_path = _base_fit_candidate(
                        base_run_root, output_root, detector, tile_id
                    )
                    with np.load(base_fit_path, allow_pickle=False) as archive:
                        base_fit = {name: np.asarray(archive[name]) for name in archive.files}
                    data = detector_data[detector][tile_id]
                    if _base_fit_matches_statistics(base_fit, stats_path, data):
                        (
                            history,
                            rate,
                            node_history,
                            node_rates,
                            records,
                            fit_values,
                        ) = _reuse_base_history(base_fit, data)
                        curves = None
                        history_update_method = (
                            "unchanged-statistics base-fit reuse; no lookup refit"
                        )
                        reused_base_fit_tiles += 1
                    else:
                        thickness, curves, _lookup_cache = load_or_build_h5_lookup(
                            output_root,
                            optics_fingerprint,
                            model,
                            reader.core,
                            detector,
                            tile_id,
                            thickness_grid,
                            source_cache_roots=(base_run_root,),
                        )
                        (
                            history,
                            rate,
                            node_history,
                            records,
                            fit_values,
                        ) = lookup_tile_history(data, curves, thickness, base_fit)
                        node_rates = np.full(
                            node_history.size, DEFAULT_RATE_MEAN_NM_DAY
                        )
                        if node_history.size > 1:
                            node_rates[1:] = np.maximum(
                                np.diff(node_history)
                                / np.maximum(np.diff(data["node_days"]), 1.0e-6),
                                1.0e-8,
                            )
                        history_update_method = "Eric-core six-LED thickness lookup"
                        h5_lookup_tiles += 1
                    fit_path = fit_root / f"tile_{tile_id:03d}" / "posterior_samples.npz"
                    write_lookup_fit_archive(
                        base_fit_path,
                        fit_path,
                        stats_path,
                        data,
                        history,
                        rate,
                        node_history,
                        node_rates,
                        base_h5_sha256,
                        records,
                        history_update_method=history_update_method,
                    )
                    if curves is None:
                        for filename in (
                            "tile_group_predictions.csv",
                            "tile_rate_posterior.csv",
                        ):
                            _copy_file_if_needed(
                                base_fit_path.parent / filename,
                                fit_path.parent / filename,
                            )
                    else:
                        write_group_predictions(
                            fit_path.parent / "tile_group_predictions.csv",
                            detector,
                            tile_id,
                            data,
                            history,
                            {"thickness_nm": thickness, "log_response": curves},
                            fit_values,
                        )
                        write_tile_rate_summary(
                            fit_path.parent / "tile_rate_posterior.csv",
                            detector,
                            tile_id,
                            data,
                            rate,
                        )
                    prior_ar = base_fit_path.parent / "tile_ar_posterior.csv"
                    if prior_ar.is_file():
                        _copy_file_if_needed(prior_ar, fit_path.parent / prior_ar.name)
                    trajectories[detector][int(tile_id)] = {
                        "history": history,
                        "rates": rate,
                        "days": np.asarray(data["days"], dtype=float),
                        "exposure_times": np.asarray(data["exposure_times"]).astype(str),
                        "led": np.asarray(data["led"], dtype=int),
                        "repeat": np.asarray(data["repeat"], dtype=int),
                        "program": np.asarray(data["program"], dtype=int),
                        "groups": np.asarray(data["groups"], dtype=int),
                        "post_reset_tile_log_offset": fit_values[
                            "post_reset_tile_log_offset"
                        ],
                        "post_third_decon_tile_log_offset": fit_values[
                            "post_third_decon_tile_log_offset"
                        ],
                    }
                    for record in _tile_lookup_records(records, detector, tile_id):
                        lookup_writer.writerow(record)
                    total_history_nodes += len(records)
                    if curves is not None:
                        total_lookup_nodes += len(records)
                        boundary_hits += sum(
                            record["thickness_nm"] <= thickness[0] + 1.0e-8
                            or record["thickness_nm"] >= thickness[-1] - 1.0e-8
                            for record in records
                        )
                if (tile_id + 1) % 64 == 0:
                    print(
                        f"{detector}: updated lookup histories for {tile_id + 1}/256 tiles",
                        flush=True,
                    )

                relative_root = fit_root.parent
                relative_runner.write_detector_rate_posterior(fit_root, relative_root)
                _write_detector_mode(
                    relative_root / "relative_response_mode.json",
                    detector,
                    statistics_dir,
                    pinned_relative_response_cache(detector),
                    base_h5_sha256,
                )
                from plot_wfi_independent_production import save_detector_plots

                save_detector_plots(
                    detector,
                    root=statistics_dir,
                    reset_day=float(detector_data[detector][0]["reset_day"]),
                    output_root=relative_root,
                    statistics_path=corrected_statistics_path(
                        statistics_dir, detector, TILE_SIZE_PIXELS
                    ),
                    tile_size=TILE_SIZE_PIXELS,
                )

    global_plot_outputs = _run_focal_plane_and_ice_trend_plots(
        statistics_dir,
        paths["plots"],
        {
            detector: fit_root.parent
            for detector, fit_root in output_fit_roots.items()
        },
        reset_day=reference_data["reset_day"],
        third_decon_day=reference_data.get("third_decon_day"),
    )

    update_metadata = {
        "method": "reuse unchanged base fits or H5 optical-state six-LED lookup",
        "base_h5": str(base_h5),
        "base_h5_sha256": base_h5_sha256,
        "optical_lookup_fingerprint": optics_fingerprint,
        "statistics_dir": str(statistics_dir),
        "detectors": args.detectors,
        "rate_prior_mean_nm_day": DEFAULT_RATE_MEAN_NM_DAY,
        "rate_prior_sigma_log": DEFAULT_RATE_SIGMA_LOG,
        "led_offsets": "reused from base per-tile posterior archives",
    }
    output_bundle_root = paths["bundle_root"]
    output_base_bundle = update_bundle_trajectories(
        base_bundle_path,
        output_bundle_root / "wfi_all_2nm_rank16.npz",
        trajectories,
        update_metadata,
    )
    output_modulated_bundle = update_bundle_trajectories(
        modulated_bundle_path,
        output_bundle_root / "wfi_all_2nm_rank16_modulated.npz",
        trajectories,
        update_metadata,
    )

    updated_bundle = bundle_io.load_bundle(output_modulated_bundle)
    history_path, factorization, _parsed_mjd, _episodes, gray_hours, time_shift = (
        make_history_products(
            model,
            base_h5,
            output_root,
            output_modulated_bundle,
            updated_bundle,
        )
    )
    output_h5 = paths["h5_root"] / "joint_modulated_v2.h5"
    output_h5 = update_reader_h5(
        base_h5,
        output_h5,
        model,
        updated_bundle,
        factorization,
        np.asarray(updated_bundle["observation_start_time"]).astype(str),
        gray_hours,
        time_shift,
    )
    summary = {
        "method": "reuse base MAP histories when statistics match exactly; otherwise H5 six-LED lookup",
        "base_h5": str(base_h5),
        "base_h5_sha256": base_h5_sha256,
        "optical_lookup_fingerprint": optics_fingerprint,
        "statistics_dir": str(statistics_dir),
        "detectors": args.detectors,
        "tile_history_count": int(updated_bundle["tile_ids"].size),
        "observation_epoch_count": int(updated_bundle["observation_days"].size),
        "lookup_thickness_step_nm": float(np.median(np.diff(thickness_grid))),
        "rate_prior_mean_nm_day": DEFAULT_RATE_MEAN_NM_DAY,
        "rate_prior_sigma_log": DEFAULT_RATE_SIGMA_LOG,
        "led_log_offsets_reused": True,
        "lookup_nodes": int(total_lookup_nodes),
        "history_nodes": int(total_history_nodes),
        "base_fit_reused_tiles": int(reused_base_fit_tiles),
        "h5_lookup_tiles": int(h5_lookup_tiles),
        "lookup_grid_boundary_hits": int(boundary_hits),
        "full_history_reference": str(history_path.resolve()),
        "base_bundle": str(output_base_bundle.resolve()),
        "modulated_bundle": str(output_modulated_bundle.resolve()),
        "final_h5": str(output_h5.resolve()),
        "h5_rank1_projection_metrics": factorization["rank1"]["metrics"],
        "focal_plane_and_ice_trend_plots": global_plot_outputs,
    }
    report_root = output_bundle_root
    (report_root / "ice_history_update_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    validation = dict(summary)
    validation_path = _write_validation(
        base_h5, output_h5, base_h5_sha256, history_path, validation
    )
    _write_updated_manifest(
        output_root,
        base_h5,
        statistics_dir,
        output_h5,
        history_path,
        summary,
    )
    print(
        json.dumps({**summary, "validation": str(validation_path)}, indent=2),
        flush=True,
    )


def _write_detector_mode(path, detector, stats_dir, output_cache_root, h5_sha):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "detector": detector,
                "fit_mode": "h5_response_lookup_ice_history_update",
                "statistics": portable_path_string(stats_dir),
                "response_cache_root": portable_path_string(output_cache_root),
                "base_h5_sha256": h5_sha,
                "tile_size": TILE_SIZE_PIXELS,
                "third_decon_day": THIRD_DECON_DAY,
                "fit_led_log_offsets": True,
                "ice_history_method": (
                    "H5 six-LED lookup with independent 48-hour log-rate spline "
                    "knots and reset-boundary knots"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_spectral_caches(base_run_root, output_root):
    source = Path(base_run_root) / "production/caches/spectral_response/relative_response"
    target = Path(output_root) / "production/caches/spectral_response/relative_response"
    if not source.is_dir() or source.resolve() == target.resolve():
        return
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        target_path = target / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        elif not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)


def _legacy_main(argv=None):
    args = parse_args(argv)
    base_h5 = args.base_h5
    statistics_dir = args.statistics_dir
    output_root = args.output_path
    bundle_root = find_bundle_root(base_h5)
    base_run_root = find_base_run_root(
        base_h5, bundle_root, explicit=args.base_run_root
    )
    base_h5_sha256 = sha256_file(base_h5)

    base_bundle_path = bundle_root / "wfi_all_2nm_rank16.npz"
    modulated_bundle_path = bundle_root / "wfi_all_2nm_rank16_modulated.npz"
    if not base_bundle_path.is_file() or not modulated_bundle_path.is_file():
        raise FileNotFoundError(
            f"The base H5 bundle directory must contain both stable bundles: {bundle_root}"
        )
    base_bundle = bundle_io.load_bundle(base_bundle_path)
    modulated_bundle = bundle_io.load_bundle(modulated_bundle_path)
    thickness_grid = np.asarray(modulated_bundle["ice_thickness_nm"], dtype=float)
    if thickness_grid.size < 2 or np.any(np.diff(thickness_grid) <= 0.0):
        raise ValueError("The base modulated bundle has an invalid thickness lookup grid")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs" / "update_ice_history").mkdir(parents=True, exist_ok=True)
    output_production = output_root / "production"
    output_production.mkdir(parents=True, exist_ok=True)
    _copy_spectral_caches(base_run_root, output_root)

    detector_data = {}
    detector_tile_paths = {}
    for detector in args.detectors:
        stats_path = corrected_statistics_path(
            statistics_dir, detector, TILE_SIZE_PIXELS
        )
        if not stats_path.is_file():
            raise FileNotFoundError(stats_path)
        output_relative_root = (
            output_production / detector.lower() / "relative_response"
        )
        tile_stats_root = output_relative_root / "tile_statistics"
        detector_tile_paths[detector] = relative_runner.tile_statistics_paths(
            detector, stats_path, tile_stats_root, TILE_SIZE_PIXELS
        )
        detector_data[detector] = {}
        for tile_id, tile_path in detector_tile_paths[detector].items():
            detector_data[detector][int(tile_id)] = production.load_tile_statistics(
                tile_path, third_decon_day=THIRD_DECON_DAY
            )
        complete_tile_ids(
            detector_tile_paths[detector], label=f"{detector} statistics tile IDs"
        )

    reference_data = detector_data["WFI01"][0]
    rate_control_days = rate_spline_knot_days(
        reference_data, spacing_days=args.rate_knot_spacing_hours / 24.0
    )
    for detector in args.detectors:
        for tile_id, data in detector_data[detector].items():
            for name in (
                "days", "led", "repeat", "program", "groups", "exposure_times"
            ):
                if not np.array_equal(
                    np.asarray(reference_data[name]), np.asarray(data[name])
                ):
                    raise ValueError(
                        f"{detector}/tile {tile_id}: input observation axis differs in {name}"
                    )

    lookup_summaries = []
    trajectories = {detector: {} for detector in DETECTORS}
    output_fit_root_by_detector = {}
    with reader_loader.load_eric_v3_reader(
        reader_loader.DEFAULT_SOURCE_ZIP
    ) as reader:
        base_model = reader.IceResponseModel(base_h5)
        optics_fingerprint = h5_optical_fingerprint(base_model)
        for detector in args.detectors:
            fit_root = output_production / detector.lower() / "relative_response" / "fits"
            output_fit_root_by_detector[detector] = fit_root
            for tile_id, data in detector_data[detector].items():
                source_fit = _base_fit_candidate(
                    base_run_root, output_root, detector, tile_id
                )
                with np.load(source_fit, allow_pickle=False) as source:
                    base_archive = {
                        name: np.asarray(source[name]) for name in source.files
                    }
                thickness, curves, _lookup_path = load_or_build_h5_lookup(
                    output_root,
                    optics_fingerprint,
                    base_model,
                    reader.core,
                    detector,
                    tile_id,
                    thickness_grid,
                )
                history, rates, node_history, records, fit_values = lookup_tile_history(
                    data, curves, thickness, base_archive
                )
                source_stats = detector_tile_paths[detector][tile_id]
                output_fit = fit_root / f"tile_{tile_id:03d}" / "posterior_samples.npz"
                write_lookup_fit_archive(
                    source_fit,
                    output_fit,
                    source_stats,
                    data,
                    history,
                    rates,
                    node_history,
                    np.maximum(
                        np.r_[
                            DEFAULT_RATE_MEAN_NM_DAY,
                            np.diff(node_history)
                            / np.maximum(np.diff(data["node_days"]), 1.0e-6),
                        ],
                        1.0e-8,
                    ),
                    base_h5_sha256,
                    records,
                )
                write_group_predictions(
                    output_fit.parent / "tile_group_predictions.csv",
                    detector,
                    tile_id,
                    data,
                    history,
                    {"thickness_nm": thickness, "log_response": curves},
                    fit_values,
                )
                write_tile_rate_summary(
                    output_fit.parent / "tile_rate_posterior.csv",
                    detector,
                    tile_id,
                    data,
                    rates,
                )
                base_ar = source_fit.parent / "tile_ar_posterior.csv"
                if base_ar.is_file():
                    _copy_file_if_needed(base_ar, output_fit.parent / base_ar.name)
                trajectories[detector][int(tile_id)] = {
                    "history": history,
                    "rates": rates,
                    "days": np.asarray(data["days"], dtype=float),
                    "exposure_times": np.asarray(data["exposure_times"]).astype(str),
                    "led": np.asarray(data["led"], dtype=int),
                    "repeat": np.asarray(data["repeat"], dtype=int),
                    "program": np.asarray(data["program"], dtype=int),
                    "groups": np.asarray(data["groups"], dtype=int),
                    "post_reset_tile_log_offset": fit_values[
                        "post_reset_tile_log_offset"
                    ],
                    "post_third_decon_tile_log_offset": fit_values[
                        "post_third_decon_tile_log_offset"
                    ],
                }
                lookup_summaries.extend(
                    {"detector": detector, "tile_id": int(tile_id), **record}
                    for record in records
                )

    for detector in args.detectors:
        fit_root = output_fit_root_by_detector[detector]
        relative_root = fit_root.parent
        relative_runner.write_detector_rate_posterior(fit_root, relative_root)
        _write_detector_mode(
            relative_root / "relative_response_mode.json",
            detector,
            statistics_dir,
            pinned_relative_response_cache(detector),
            base_h5_sha256,
        )

    metadata_update = {
        "method": "H5 six-LED response lookup",
        "base_h5_sha256": base_h5_sha256,
        "optical_lookup_fingerprint": optics_fingerprint,
        "statistics_dir": str(statistics_dir),
        "rate_prior_mean_nm_day": DEFAULT_RATE_MEAN_NM_DAY,
        "rate_prior_sigma_log": DEFAULT_RATE_SIGMA_LOG,
        "led_offsets": "reused from base per-tile fits",
    }
    output_bundle_root = output_production / "wavelength_response_bundle"
    output_base_bundle = update_bundle_trajectories(
        base_bundle_path,
        output_bundle_root / "wfi_all_2nm_rank16.npz",
        trajectories,
        metadata_update,
    )
    output_modulated_bundle = update_bundle_trajectories(
        modulated_bundle_path,
        output_bundle_root / "wfi_all_2nm_rank16_modulated.npz",
        trajectories,
        metadata_update,
    )

    updated_bundle = bundle_io.load_bundle(output_modulated_bundle)
    histories = np.asarray(updated_bundle["tile_ice_thickness"], dtype=float)
    rates = np.asarray(updated_bundle["tile_instantaneous_rate_nm_day"], dtype=float)
    reference_times = np.asarray(updated_bundle["observation_start_time"]).astype(str)
    history_path, factorization, parsed_mjd, episodes, gray_hours, time_shift = (
        make_history_products(
            base_model,
            base_h5,
            output_root,
            output_modulated_bundle,
            updated_bundle,
        )
    )
    output_h5 = output_paths(output_root, RELATIVE_RESPONSE)["h5_root"] / "joint_modulated_v2.h5"
    output_h5 = update_reader_h5(
        base_h5,
        output_h5,
        base_model,
        updated_bundle,
        factorization,
        reference_times,
        gray_hours,
        time_shift,
    )

    report = {
        "method": "H5 six-LED lookup over all corrected-statistics epochs",
        "base_h5_sha256": base_h5_sha256,
        "optical_lookup_fingerprint": optics_fingerprint,
        "statistics_dir": str(statistics_dir),
        "detectors": args.detectors,
        "tile_history_count": int(histories.shape[0]),
        "observation_epoch_count": int(histories.shape[1]),
        "lookup_thickness_step_nm": float(np.median(np.diff(thickness_grid))),
        "rate_prior_mean_nm_day": DEFAULT_RATE_MEAN_NM_DAY,
        "rate_prior_sigma_log": DEFAULT_RATE_SIGMA_LOG,
        "led_log_offsets_reused": True,
        "updated_repeat_count": int(len(lookup_summaries)),
        "full_history_reference": str(history_path.resolve()),
        "final_h5": str(output_h5.resolve()),
        "base_bundle": str(output_base_bundle.resolve()),
        "modulated_bundle": str(output_modulated_bundle.resolve()),
    }
    output_bundle_root.mkdir(parents=True, exist_ok=True)
    (output_bundle_root / "ice_history_update_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lookup_csv = output_bundle_root / "ice_history_lookup_nodes.csv"
    with lookup_csv.open("w", newline="") as handle:
        fields = list(lookup_summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lookup_summaries)
    validation = dict(report)
    validation_path = _write_validation(
        base_h5, output_h5, base_h5_sha256, history_path, validation
    )
    _write_updated_manifest(
        output_root,
        base_h5,
        statistics_dir,
        output_h5,
        history_path,
        report,
    )
    print(json.dumps({**report, "validation": str(validation_path)}, indent=2), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-h5", type=Path, required=True)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of independent tile MAP fits solved in one accelerator batch.",
    )
    parser.add_argument("--optimizer-iterations", type=int, default=160)
    parser.add_argument("--rate-knot-spacing-hours", type=float, default=48.0)
    parser.add_argument(
        "--detector",
        action="append",
        required=True,
        choices=DETECTORS,
        help="Detector to update; repeat explicitly for WFI01 through WFI18.",
    )
    args = parser.parse_args(argv)
    args.detectors = list(dict.fromkeys(args.detector))
    if set(args.detectors) != set(DETECTORS):
        parser.error("Pass all 18 detectors explicitly with repeated --detector flags")
    if not args.base_h5.is_file():
        parser.error(f"Base H5 does not exist: {args.base_h5}")
    if not args.statistics_dir.is_dir():
        parser.error(f"Statistics directory does not exist: {args.statistics_dir}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.optimizer_iterations < 1:
        parser.error("--optimizer-iterations must be positive")
    if not np.isfinite(args.rate_knot_spacing_hours) or args.rate_knot_spacing_hours <= 0.0:
        parser.error("--rate-knot-spacing-hours must be positive and finite")
    return args


def main_streaming(argv=None):
    """Fit independent tile histories from the supplied H5 and statistics only."""
    args = parse_args(argv)
    base_h5 = args.base_h5.resolve()
    statistics_dir = args.statistics_dir.resolve()
    output_root = args.output_path.resolve()
    run_started = time.perf_counter()
    base_h5_sha256 = sha256_file(base_h5)

    # These bundles are output scaffolds for preserving the standard portable
    # spectral products. The fit itself reads its optical state only from H5.
    bundle_root = find_bundle_root(base_h5)
    base_bundle_path = bundle_root / "wfi_all_2nm_rank16.npz"
    modulated_bundle_path = bundle_root / "wfi_all_2nm_rank16_modulated.npz"
    if not base_bundle_path.is_file() or not modulated_bundle_path.is_file():
        raise FileNotFoundError(f"Missing standard output bundle scaffolds in {bundle_root}")
    base_bundle = bundle_io.load_bundle(base_bundle_path)
    modulated_bundle = bundle_io.load_bundle(modulated_bundle_path)
    thickness_grid = np.arange(
        0.0,
        500.0 + 0.5 * DEFAULT_LOOKUP_STEP_NM,
        DEFAULT_LOOKUP_STEP_NM,
        dtype=float,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_root, RELATIVE_RESPONSE)
    for directory in (paths["logs"], paths["production"], paths["bundle_root"]):
        directory.mkdir(parents=True, exist_ok=True)

    detector_data = {}
    detector_tile_paths = {}
    for detector in args.detectors:
        statistics_path = corrected_statistics_path(
            statistics_dir, detector, TILE_SIZE_PIXELS
        )
        if not statistics_path.is_file():
            raise FileNotFoundError(statistics_path)
        relative_root = paths["production"] / detector.lower() / "relative_response"
        detector_tile_paths[detector] = relative_runner.tile_statistics_paths(
            detector,
            statistics_path,
            relative_root / "tile_statistics",
            TILE_SIZE_PIXELS,
        )
        complete_tile_ids(
            detector_tile_paths[detector], label=f"{detector} statistics tile IDs"
        )
        detector_data[detector] = {}
        for tile_id, tile_stats_path in detector_tile_paths[detector].items():
            data = production.load_tile_statistics(
                tile_stats_path, third_decon_day=THIRD_DECON_DAY
            )
            data["statistics_path"] = str(tile_stats_path)
            detector_data[detector][int(tile_id)] = data

    reference_data = detector_data["WFI01"][0]
    rate_control_days = rate_spline_knot_days(
        reference_data, spacing_days=args.rate_knot_spacing_hours / 24.0
    )
    for detector in args.detectors:
        for tile_id, data in detector_data[detector].items():
            for name in (
                "days", "led", "repeat", "program", "groups", "exposure_times"
            ):
                if not np.array_equal(reference_data[name], data[name]):
                    raise ValueError(
                        f"{detector}/tile {tile_id}: statistics observation layout differs in {name}"
                    )

    trajectories = {detector: {} for detector in DETECTORS}
    output_fit_roots = {}
    lookup_records_path = paths["bundle_root"] / "ice_history_lookup_nodes.csv"
    total_history_nodes = 0
    boundary_hits = 0
    lookup_seconds = 0.0
    optimizer_seconds = 0.0
    detector_plot_seconds = 0.0

    jax, jnp = production.require_jax()
    jax.config.update("jax_enable_x64", True)
    with reader_loader.load_eric_v3_reader(reader_loader.DEFAULT_SOURCE_ZIP) as reader:
        model = reader.IceResponseModel(str(base_h5))
        optics_fingerprint = h5_optical_fingerprint(model)
        lookup_root = (
            paths["production"] / "caches" / DEFAULT_LOOKUP_CACHE
            / optics_fingerprint[:16]
        )
        with lookup_records_path.open("w", newline="") as lookup_csv:
            lookup_fields = (
                "detector", "tile_id", "repeat", "day", "thickness_nm",
                "rate_nm_day", "objective_cost", "candidate_count", "led_count",
            )
            lookup_writer = csv.DictWriter(lookup_csv, fieldnames=lookup_fields)
            lookup_writer.writeheader()

            for detector in args.detectors:
                fit_root = (
                    paths["production"] / detector.lower()
                    / "relative_response" / "fits"
                )
                relative_root = fit_root.parent
                output_fit_roots[detector] = fit_root
                tile_ids = sorted(detector_data[detector])
                data_list = [detector_data[detector][tile_id] for tile_id in tile_ids]
                initial_values_list = []
                prior_centers_list = []
                prior_sigmas_list = []
                curve_entries = []
                lookup_paths = []
                for tile_id, data in zip(tile_ids, data_list, strict=True):
                    values = production.initial_values_for_data(
                        data,
                        constant_rate=False,
                        fit_led_log_offsets=True,
                    )
                    values, h5_thickness_seed, _h5_rate_seed = initialize_tile_from_h5(
                        model, detector, tile_id, data, values
                    )
                    initial_values_list.append(values)
                    thickness_names, _offset_names = production.reset_site_names(data)
                    prior_centers = {}
                    prior_sigmas = {}
                    reset_days = production.reset_days_for_data(data)
                    for stage, name in enumerate(thickness_names):
                        if name == "third_decon_initial_thickness_log_nm":
                            starts = np.flatnonzero(
                                np.asarray(data["node_days"], dtype=float)
                                >= float(data["third_decon_day"])
                            )
                            center = (
                                float(h5_thickness_seed[int(starts[0])])
                                if starts.size
                                else production.INITIAL_THICKNESS_LOG_MEDIAN_NM
                            )
                            prior_centers[name] = np.log(max(center, 1.0e-3))
                            # The H5's per-tile decon state replaces the external
                            # warmup center; retain the validated tight log prior.
                            prior_sigmas[name] = 0.1
                        else:
                            prior_centers[name] = np.log(
                                production.INITIAL_THICKNESS_LOG_MEDIAN_NM
                            )
                            prior_sigmas[name] = production.INITIAL_THICKNESS_LOG_SIGMA
                    prior_centers_list.append(prior_centers)
                    prior_sigmas_list.append(prior_sigmas)
                    lookup_started = time.perf_counter()
                    grid, curves, cache_path = load_or_build_h5_lookup(
                        output_root,
                        optics_fingerprint,
                        model,
                        reader.core,
                        detector,
                        tile_id,
                        thickness_grid,
                    )
                    lookup_seconds += time.perf_counter() - lookup_started
                    curve_entries.append(
                        {"thickness_nm": grid, "log_response": curves}
                    )
                    lookup_paths.append(cache_path)

                for batch_start in range(0, len(tile_ids), args.batch_size):
                    batch_stop = min(batch_start + args.batch_size, len(tile_ids))
                    batch_data = data_list[batch_start:batch_stop]
                    optimizer_started = time.perf_counter()
                    batch_values, batch_diagnostics = fit_h5_tile_batch(
                        batch_data,
                        curve_entries[batch_start:batch_stop],
                        initial_values_list[batch_start:batch_stop],
                        jax,
                        jnp,
                        led_log_offset_prior_sigma_log=0.01,
                        max_iterations=args.optimizer_iterations,
                        rate_knot_days=rate_control_days,
                        thickness_prior_centers_list=prior_centers_list[
                            batch_start:batch_stop
                        ],
                        thickness_prior_sigmas_list=prior_sigmas_list[
                            batch_start:batch_stop
                        ],
                    )
                    optimizer_seconds += time.perf_counter() - optimizer_started
                    print(
                        f"{detector}: fitted independent H5 histories "
                        f"{batch_stop}/{len(tile_ids)} tiles",
                        flush=True,
                    )
                    for local_index, tile_id in enumerate(tile_ids[batch_start:batch_stop]):
                        global_index = batch_start + local_index
                        data = batch_data[local_index]
                        values = batch_values[local_index]
                        diagnostic = batch_diagnostics[local_index]
                        curve = curve_entries[global_index]
                        cache_path = lookup_paths[global_index]
                        state = relative_model.relative_state(
                            data,
                            values,
                            {
                                "grid_nm": curve["thickness_nm"],
                                "relative_log_response": curve["log_response"],
                            },
                            jax,
                            jnp,
                        )
                        fit_path = (
                            fit_root / f"tile_{tile_id:03d}" / "posterior_samples.npz"
                        )
                        relative_fit.write_outputs(
                            fit_path.parent,
                            data,
                            values,
                            state,
                            diagnostic,
                            cache_path,
                            jax,
                            jnp,
                            fit_led_log_offsets=True,
                            led_log_offset_prior_sigma_log=0.01,
                            fit_mode="h5_only_batched_ice_history_update",
                        )
                        history = np.asarray(state["ice"], dtype=float).reshape(-1)
                        rates = np.asarray(state["rates"], dtype=float).reshape(-1)
                        node_history = np.asarray(state["node_ice"], dtype=float).reshape(-1)
                        node_rates = np.asarray(state["node_rates"], dtype=float).reshape(-1)
                        led_offsets = np.asarray(values["led_log_offsets"], dtype=float).reshape(-1)
                        reset_offsets = np.asarray(
                            values["post_reset_tile_log_offset"], dtype=float
                        ).reshape(-1)
                        third_offsets = (
                            np.asarray(values["post_third_decon_tile_log_offset"], dtype=float)
                            .reshape(-1)
                            if "post_third_decon_tile_log_offset" in values
                            else None
                        )
                        trajectories[detector][int(tile_id)] = {
                            "history": history,
                            "rates": rates,
                            "days": np.asarray(data["days"], dtype=float),
                            "exposure_times": np.asarray(data["exposure_times"]).astype(str),
                            "led": np.asarray(data["led"], dtype=int),
                            "repeat": np.asarray(data["repeat"], dtype=int),
                            "program": np.asarray(data["program"], dtype=int),
                            "groups": np.asarray(data["groups"], dtype=int),
                            "post_reset_tile_log_offset": float(reset_offsets[0]),
                            "post_third_decon_tile_log_offset": (
                                None if third_offsets is None else float(third_offsets[0])
                            ),
                        }
                        for repeat, day, thickness_nm, rate_nm_day in zip(
                            data["repeat_values"],
                            data["node_days"],
                            node_history,
                            node_rates,
                            strict=True,
                        ):
                            lookup_writer.writerow(
                                {
                                    "detector": detector,
                                    "tile_id": int(tile_id),
                                    "repeat": int(repeat),
                                    "day": float(day),
                                    "thickness_nm": float(thickness_nm),
                                    "rate_nm_day": float(rate_nm_day),
                                    "objective_cost": float(diagnostic["residual_cost"]),
                                    "candidate_count": 0,
                                    "led_count": 6,
                                }
                            )
                        total_history_nodes += int(node_history.size)
                        boundary_hits += int(
                            np.any(history <= thickness_grid[0] + 1.0e-8)
                            or np.any(history >= thickness_grid[-1] - 1.0e-8)
                        )

                relative_runner.write_detector_rate_posterior(fit_root, relative_root)
                _write_detector_mode(
                    relative_root / "relative_response_mode.json",
                    detector,
                    statistics_dir,
                    lookup_root / detector.lower(),
                    base_h5_sha256,
                )
                from plot_wfi_independent_production import save_detector_plots

                detector_plot_started = time.perf_counter()
                save_detector_plots(
                    detector,
                    root=statistics_dir,
                    reset_day=float(detector_data[detector][0]["reset_day"]),
                    output_root=relative_root,
                    statistics_path=corrected_statistics_path(
                        statistics_dir, detector, TILE_SIZE_PIXELS
                    ),
                    tile_size=TILE_SIZE_PIXELS,
                )
                detector_plot_seconds += time.perf_counter() - detector_plot_started

    global_plot_started = time.perf_counter()
    global_plot_outputs = _run_focal_plane_and_ice_trend_plots(
        statistics_dir,
        paths["plots"],
        {
            detector: fit_root.parent
            for detector, fit_root in output_fit_roots.items()
        },
        reset_day=reference_data["reset_day"],
        third_decon_day=reference_data.get("third_decon_day"),
    )
    global_plot_seconds = time.perf_counter() - global_plot_started

    update_metadata = {
        "method": "H5-only response curves with batched independent rate-node MAP",
        "base_h5": str(base_h5),
        "base_h5_sha256": base_h5_sha256,
        "optical_lookup_fingerprint": optics_fingerprint,
        "statistics_dir": str(statistics_dir),
        "detectors": args.detectors,
        "optimizer": "batched damped Gauss-Newton, block-independent per tile",
        "optimizer_iterations": args.optimizer_iterations,
        "optimizer_batch_size": args.batch_size,
        "rate_spline_knot_spacing_hours": float(args.rate_knot_spacing_hours),
        "rate_spline_knot_count": int(rate_control_days.size),
        "rate_prior_mean_nm_day": DEFAULT_RATE_MEAN_NM_DAY,
        "rate_prior_sigma_log": DEFAULT_RATE_SIGMA_LOG,
        "led_offsets": "fit from corrected statistics with log sigma 0.01",
    }
    output_bundle_root = paths["bundle_root"]
    output_base_bundle = update_bundle_trajectories(
        base_bundle_path,
        output_bundle_root / "wfi_all_2nm_rank16.npz",
        trajectories,
        update_metadata,
    )
    output_modulated_bundle = update_bundle_trajectories(
        modulated_bundle_path,
        output_bundle_root / "wfi_all_2nm_rank16_modulated.npz",
        trajectories,
        update_metadata,
    )
    updated_bundle = bundle_io.load_bundle(output_modulated_bundle)
    history_path, factorization, _parsed_mjd, _episodes, gray_hours, time_shift = (
        make_history_products(
            model,
            base_h5,
            output_root,
            output_modulated_bundle,
            updated_bundle,
        )
    )
    output_h5 = paths["h5_root"] / "joint_modulated_v2.h5"
    output_h5 = update_reader_h5(
        base_h5,
        output_h5,
        model,
        updated_bundle,
        factorization,
        np.asarray(updated_bundle["observation_start_time"]).astype(str),
        gray_hours,
        time_shift,
    )
    summary = {
        "method": "H5-only optical lookup with independent batched MAP histories",
        "base_h5": str(base_h5),
        "base_h5_sha256": base_h5_sha256,
        "optical_lookup_fingerprint": optics_fingerprint,
        "statistics_dir": str(statistics_dir),
        "detectors": args.detectors,
        "tile_history_count": int(updated_bundle["tile_ids"].size),
        "observation_epoch_count": int(updated_bundle["observation_days"].size),
        "lookup_thickness_step_nm": float(DEFAULT_LOOKUP_STEP_NM),
        "optimizer": "batched damped Gauss-Newton with independent tile blocks",
        "optimizer_iterations": int(args.optimizer_iterations),
        "optimizer_batch_size": int(args.batch_size),
        "rate_spline_knot_spacing_hours": float(args.rate_knot_spacing_hours),
        "rate_spline_knot_count": int(rate_control_days.size),
        "h5_lookup_seconds": float(lookup_seconds),
        "optimizer_seconds": float(optimizer_seconds),
        "per_detector_plot_seconds": float(detector_plot_seconds),
        "global_plot_seconds": float(global_plot_seconds),
        "total_run_seconds": float(time.perf_counter() - run_started),
        "led_log_offsets_fitted": True,
        "led_log_offset_prior_sigma_log": 0.01,
        "history_nodes": int(total_history_nodes),
        "lookup_grid_boundary_hits": int(boundary_hits),
        "full_history_reference": str(history_path.resolve()),
        "base_bundle": str(output_base_bundle.resolve()),
        "modulated_bundle": str(output_modulated_bundle.resolve()),
        "final_h5": str(output_h5.resolve()),
        "h5_rank1_projection_metrics": factorization["rank1"]["metrics"],
        "focal_plane_and_ice_trend_plots": global_plot_outputs,
    }
    (output_bundle_root / "ice_history_update_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    validation_path = _write_validation(
        base_h5, output_h5, base_h5_sha256, history_path, dict(summary)
    )
    _write_updated_manifest(
        output_root,
        base_h5,
        statistics_dir,
        output_h5,
        history_path,
        summary,
    )
    print(
        json.dumps({**summary, "validation": str(validation_path)}, indent=2),
        flush=True,
    )


def _base_fit_candidate(base_run_root, output_root, detector, tile_id):
    relative = Path("production") / detector.lower() / "relative_response" / "fits"
    candidates = (
        Path(output_root) / relative / f"tile_{tile_id:03d}" / "posterior_samples.npz",
        Path(base_run_root) / relative / f"tile_{tile_id:03d}" / "posterior_samples.npz",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No base per-tile fit for {detector}/tile {tile_id}; LED offsets and "
        f"initial histories were not found in {candidates[0]} or {candidates[1]}"
    )


def _write_lookup_mode(path, detector, statistics_dir, cache_root, base_h5_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "detector": detector,
                "fit_mode": "h5_response_lookup_ice_history_update",
                "statistics": portable_path_string(statistics_dir),
                "response_cache_root": portable_path_string(cache_root),
                "base_h5_sha256": base_h5_sha256,
                "ice_history_method": "six-LED grid lookup with rate/reset continuity",
                "fit_led_log_offsets": "reused from base per-tile fits",
                "tile_size": TILE_SIZE_PIXELS,
                "third_decon_day": THIRD_DECON_DAY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _unused_duplicate_main(argv=None):
    args = parse_args(argv)
    base_h5 = Path(args.base_h5).resolve()
    statistics_dir = Path(args.statistics_dir).resolve()
    output_root = Path(args.output_path).resolve()
    bundle_root = find_bundle_root(base_h5)
    base_run_root = find_base_run_root(base_h5, bundle_root, args.base_run_root)
    base_h5_sha256 = sha256_file(base_h5)
    base_bundle_path = bundle_root / "wfi_all_2nm_rank16.npz"
    modulated_bundle_path = bundle_root / "wfi_all_2nm_rank16_modulated.npz"
    for path in (base_bundle_path, modulated_bundle_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_bundle = bundle_io.load_bundle(base_bundle_path)
    modulated_bundle = bundle_io.load_bundle(modulated_bundle_path)
    thickness_grid = np.arange(
        0.0,
        args.lookup_max_thickness_nm + 0.5 * args.lookup_step_nm,
        args.lookup_step_nm,
        dtype=float,
    )
    if thickness_grid.size < 2:
        raise ValueError("Lookup thickness grid must contain at least two points")
    optics_fingerprint = h5_optical_fingerprint

    paths = output_paths(output_root, "relative_response")
    production_root = paths["production"]
    lookup_summaries = []
    trajectories = {detector: {} for detector in DETECTORS}
    fit_roots = {}
    reference_data = None

    with reader_loader.load_eric_v3_reader(reader_loader.DEFAULT_SOURCE_ZIP) as reader:
        base_model = reader.IceResponseModel(base_h5)
        optics_key = optics_fingerprint(base_model)
        for detector in args.detectors:
            statistics_path = corrected_statistics_path(
                statistics_dir, detector, TILE_SIZE_PIXELS
            )
            if not statistics_path.is_file():
                raise FileNotFoundError(statistics_path)
            fit_root = production_root / detector.lower() / "relative_response" / "fits"
            fit_roots[detector] = fit_root
            split_root = production_root / detector.lower() / "relative_response" / "tile_statistics"
            tile_paths = relative_runner.tile_statistics_paths(
                detector,
                statistics_path,
                split_root,
                TILE_SIZE_PIXELS,
            )
            complete_tiles = complete_tile_ids(tile_paths)
            if len(complete_tiles) != 256:
                raise ValueError(f"{detector}: expected 256 tiles, found {len(complete_tiles)}")

            for tile_id, tile_stats in tile_paths.items():
                data = production.load_tile_statistics(
                    tile_stats, third_decon_day=THIRD_DECON_DAY
                )
                if reference_data is None:
                    reference_data = data
                else:
                    for name in (
                        "days", "led", "repeat", "program", "groups", "exposure_times"
                    ):
                        if not np.array_equal(
                            np.asarray(reference_data[name]), np.asarray(data[name])
                        ):
                            raise ValueError(
                                f"{detector}/tile {tile_id}: observation layout differs in {name}"
                            )

                base_fit_path = _base_fit_candidate(
                    base_run_root, output_root, detector, tile_id
                )
                with np.load(base_fit_path, allow_pickle=False) as base_fit:
                    base_archive = {name: np.asarray(base_fit[name]) for name in base_fit.files}
                cache_path = load_or_build_h5_lookup(
                    output_root,
                    optics_key,
                    base_model,
                    reader.core,
                    detector,
                    tile_id,
                    thickness_grid,
                )
                with np.load(cache_path[2], allow_pickle=False) as lookup_archive:
                    lookup_curves = np.asarray(
                        lookup_archive["relative_log_response"], dtype=float
                    )
                history, rates, node_history, records, fit_values = lookup_tile_history(
                    data,
                    lookup_curves,
                    thickness_grid,
                    base_archive,
                )
                output_fit = fit_root / f"tile_{tile_id:03d}" / "posterior_samples.npz"
                write_lookup_fit_archive(
                    base_fit_path,
                    output_fit,
                    tile_stats,
                    data,
                    history,
                    rates,
                    node_history,
                    np.maximum(
                        np.r_[DEFAULT_RATE_MEAN_NM_DAY, np.diff(node_history)
                              / np.maximum(np.diff(data["node_days"]), 1.0e-6)],
                        1.0e-8,
                    ),
                    base_h5_sha256,
                    records,
                )
                write_group_predictions(
                    output_fit.parent / "tile_group_predictions.csv",
                    detector,
                    tile_id,
                    data,
                    history,
                    {"thickness_nm": thickness_grid, "log_response": lookup_curves},
                    fit_values,
                )
                write_tile_rate_summary(
                    output_fit.parent / "tile_rate_posterior.csv",
                    detector,
                    tile_id,
                    data,
                    rates,
                )
                if (base_fit_path.parent / "tile_ar_posterior.csv").is_file():
                    _copy_file_if_needed(
                        base_fit_path.parent / "tile_ar_posterior.csv",
                        output_fit.parent / "tile_ar_posterior.csv",
                    )
                trajectories[detector][int(tile_id)] = {
                    "history": history,
                    "rates": rates,
                    "days": np.asarray(data["days"], dtype=float),
                    "exposure_times": np.asarray(data["exposure_times"]).astype(str),
                    "led": np.asarray(data["led"], dtype=int),
                    "repeat": np.asarray(data["repeat"], dtype=int),
                    "program": np.asarray(data["program"], dtype=int),
                    "groups": np.asarray(data["groups"], dtype=int),
                    "post_reset_tile_log_offset": fit_values[
                        "post_reset_tile_log_offset"
                    ],
                    "post_third_decon_tile_log_offset": fit_values[
                        "post_third_decon_tile_log_offset"
                    ],
                }
                lookup_summaries.extend(
                    {
                        "detector": detector,
                        "tile_id": int(tile_id),
                        **record,
                    }
                    for record in records
                )

    if reference_data is None:
        raise ValueError("No detector statistics were processed")
    output_bundle_root = production_root / "wavelength_response_bundle"
    output_bundle_root.mkdir(parents=True, exist_ok=True)
    update_metadata = {
        "method": "H5 six-LED thickness lookup",
        "base_h5": str(base_h5),
        "base_h5_sha256": base_h5_sha256,
        "optical_lookup_fingerprint": optics_key,
        "statistics_dir": str(statistics_dir),
        "detectors": args.detectors,
        "rate_prior_mean_nm_day": DEFAULT_RATE_MEAN_NM_DAY,
        "rate_prior_sigma_log": DEFAULT_RATE_SIGMA_LOG,
        "led_log_offsets": "reused from base per-tile posterior archives",
    }
    output_base_bundle = update_bundle_trajectories(
        base_bundle_path,
        output_bundle_root / "wfi_all_2nm_rank16.npz",
        trajectories,
        update_metadata,
    )
    output_modulated_bundle = update_bundle_trajectories(
        modulated_bundle_path,
        output_bundle_root / "wfi_all_2nm_rank16_modulated.npz",
        trajectories,
        update_metadata,
    )

    for detector in args.detectors:
        fit_root = fit_roots[detector]
        relative_root = fit_root.parent
        relative_runner.write_detector_rate_posterior(fit_root, relative_root)
        _write_lookup_mode(
            relative_root / "relative_response_mode.json",
            detector,
            statistics_dir,
            base_run_root / "20261109/corrected_statistics_4h_thinned/production/relative_response_cache",
            base_h5_sha256,
        )

    histories = np.asarray(
        [
            trajectories[str(detector)][int(tile_id)]["history"]
            for detector, tile_id in zip(
                output_modulated_bundle.exists() and modulated_bundle["detectors"],
                modulated_bundle["tile_ids"],
                strict=True,
            )
        ],
        dtype=float,
    )
    updated_bundle = bundle_io.load_bundle(output_modulated_bundle)
    rates = np.asarray(updated_bundle["tile_instantaneous_rate_nm_day"], dtype=float)
    reference_days = np.asarray(updated_bundle["observation_days"], dtype=float)
    reference_times = np.asarray(updated_bundle["observation_start_time"]).astype(str)
    history_path, factorization, parsed_mjd, episodes, gray_hours, time_shift = (
        make_history_products(
            base_model,
            base_h5,
            output_root,
            output_modulated_bundle,
            updated_bundle,
        )
    )
    output_h5 = output_paths(output_root, "relative_response")["h5_root"] / "joint_modulated_v2.h5"
    output_h5 = update_reader_h5(
        base_h5,
        output_h5,
        base_model,
        updated_bundle,
        factorization,
        reference_times,
        gray_hours,
        time_shift,
    )
    lookup_cache_file_count = sum(
        len(list((production_root / "caches" / DEFAULT_LOOKUP_CACHE / optics_key / detector.lower()).glob("tile_*/ice_response_lookup.npz")))
        for detector in args.detectors
    )
    summary = {
        "method": "cached H5 six-LED thickness lookup",
        "base_h5_sha256": base_h5_sha256,
        "optical_lookup_fingerprint": optics_key,
        "statistics_dir": str(statistics_dir),
        "detectors": args.detectors,
        "tile_histories": int(sum(len(trajectories[d]) for d in args.detectors)),
        "data_epochs": int(reference_days.size),
        "new_lookup_nodes": int(len(lookup_summaries)),
        "lookup_cache_files": int(lookup_cache_file_count),
        "lookup_thickness_step_nm": float(np.median(np.diff(thickness_grid))),
        "led_offsets_reused": True,
        "full_history_reference": str(history_path.resolve()),
        "final_h5": str(output_h5.resolve()),
        "base_bundle": str(output_base_bundle.resolve()),
        "modulated_bundle": str(output_modulated_bundle.resolve()),
    }
    summary_path = output_bundle_root / "ice_history_update_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lookup_csv = output_bundle_root / "ice_history_lookup_nodes.csv"
    if lookup_summaries:
        with lookup_csv.open("w", newline="") as handle:
            fields = list(lookup_summaries[0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(lookup_summaries)
    validation_report = dict(summary)
    validation_path = _write_validation(
        base_h5,
        output_h5,
        base_h5_sha256,
        history_path,
        validation_report,
    )
    _write_updated_manifest(
        output_root,
        base_h5,
        statistics_dir,
        output_h5,
        history_path,
        summary,
    )
    print(json.dumps({**summary, "validation": str(validation_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main_streaming()
