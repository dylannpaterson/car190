"""Plot a NumPyro TMM tile posterior for one or all WFI SCAs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fit_wfi_tmm_numpyro import (
    ASW_OPTICAL_CONSTANTS_URL,
    QE_RBF_NODE_WAVELENGTHS_NM,
    QE_RBF_WIDTH_NM,
    PREDICTION_ROW_CHUNK_SIZE,
    early_time_scale_configuration,
)
from assess_rcs_spectral_identifiability import QE_URL, load_qe
from wfi_production_plot_annotations import (
    annotate_time_axis,
    event_legend_handles,
    non_annotation_legend_handles,
    split_time_series_at_reset,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SINGLE_SCA_OUTPUT = ROOT / "wfi01_tmm_numpyro_tile_pilot"
DEFAULT_MODEL_PATH = Path(
    "/Users/paterson.35/Documents/"
    "rgespit-comissioning-ice-simulations/data/ice_optical_constants/"
    "spectral_response_model_SCA1.ecsv"
)


def scope_label(sca_names):
    names = []
    for value in np.asarray(sca_names).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode()
        names.append(str(value))
    if len(names) == 1:
        return names[0]
    return f"{len(names)} WFI SCAs"


def sample_scope(samples):
    if "sca_names" in samples:
        return scope_label(samples["sca_names"])
    return "WFI01"


def ar_layer_count(samples):
    if "ar_layers" in samples:
        return int(np.asarray(samples["ar_layers"]).reshape(-1)[0])
    ar_mean = np.asarray(samples["ar_mean"])
    if ar_mean.shape[-1] % 3:
        raise ValueError("AR mean posterior width must be divisible by three")
    return ar_mean.shape[-1] // 3


def ar_parameter_labels(ar_layers, relative=False):
    labels = []
    for layer in range(ar_layers):
        prefix = "AR " if ar_layers == 1 else f"AR layer {layer + 1} "
        suffix = " / median" if relative else " mean"
        thickness = "scale" if ar_layers == 1 else "thickness scale"
        labels.extend(
            (
                f"{prefix}{thickness}{suffix}",
                f"{prefix}n{suffix}",
                f"{prefix}k{suffix}",
            )
        )
    return tuple(labels)


def hierarchy_sigma_labels(ar_layers, parameter_count, scope):
    if parameter_count == ar_layers:
        components = ("scale",)
    elif parameter_count == 2 * ar_layers:
        components = ("scale", "n")
    elif parameter_count == 3 * ar_layers:
        components = ("scale", "n", "k")
    else:
        raise ValueError("AR hierarchy width does not match the layer count")
    return tuple(
        f"{f'layer {layer + 1} ' if ar_layers > 1 else ''}{scope} sigma {component}"
        for layer in range(ar_layers)
        for component in components
    )


def load_group_metadata(path):
    if path.is_dir():
        candidates = sorted(path.glob("wfi*_tile_log_statistics_*.csv"))
        if not candidates:
            raise ValueError(f"No completed SCA statistics CSVs found in {path}")
        path = candidates[0]
    metadata = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            group = int(row["group"])
            metadata[group] = {
                "led": int(row["led"]),
                "repeat": int(row["repeat"]),
                "days": float(row["days"]),
            }
    if not metadata:
        raise ValueError(f"No tile statistics found in {path}")
    groups = sorted(metadata)
    if groups != list(range(len(groups))):
        raise ValueError("Tile statistics groups must span zero through the final group")
    return (
        np.asarray([metadata[group]["days"] for group in groups]),
        np.asarray([metadata[group]["led"] for group in groups]),
        np.asarray([metadata[group]["repeat"] for group in groups]),
    )


def save_trajectory_plot(path, samples, days, seed):
    thickness = np.asarray(samples["ice_thickness"], dtype=float)
    tile_thickness = (
        np.asarray(samples["tile_ice_thickness"], dtype=float)
        if "tile_ice_thickness" in samples
        else None
    )
    tile_trajectory = (
        tile_thickness is not None
        and tile_thickness.ndim == 3
        and tile_thickness.shape[-1] == len(days)
    )
    if tile_trajectory:
        tile_thickness_quantiles = np.percentile(
            tile_thickness, [16, 50, 84], axis=0
        )
        thickness = np.mean(tile_thickness, axis=1)
    elif thickness.ndim == 3:
        thickness = np.mean(thickness, axis=1)
    if thickness.ndim != 2 or thickness.shape[1] != len(days):
        raise ValueError("Ice-thickness samples do not match the tile-statistics groups")
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    sorted_thickness = thickness[:, order]
    instantaneous_rates = (
        samples["instantaneous_rate_nm_day"]
        if "instantaneous_rate_nm_day" in samples
        else None
    )
    if instantaneous_rates is not None:
        instantaneous_rates = np.asarray(instantaneous_rates, dtype=float)
    tile_rates = (
        np.asarray(samples["tile_instantaneous_rate_nm_day"], dtype=float)
        if "tile_instantaneous_rate_nm_day" in samples
        else None
    )
    tile_rate_trajectory = (
        tile_trajectory
        and tile_rates is not None
        and tile_rates.ndim == 3
        and tile_rates.shape[-1] == len(days)
    )
    has_rate_trajectory = (
        instantaneous_rates is not None
        and instantaneous_rates.ndim == 2
        and instantaneous_rates.shape == thickness.shape
    )
    quantiles = np.percentile(sorted_thickness, [16, 50, 84], axis=0)
    rng = np.random.default_rng(seed)
    draw_count = min(100, len(sorted_thickness))
    draw_indices = rng.choice(len(sorted_thickness), draw_count, replace=False)

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    if tile_trajectory:
        tile_colors = plt.get_cmap("viridis", tile_thickness_quantiles.shape[1])
        for tile_index in range(tile_thickness_quantiles.shape[1]):
            color = tile_colors(tile_index)
            axes[0].fill_between(
                sorted_days,
                tile_thickness_quantiles[0, tile_index, order],
                tile_thickness_quantiles[2, tile_index, order],
                color=color,
                alpha=0.025,
            )
            axes[0].plot(
                sorted_days,
                tile_thickness_quantiles[1, tile_index, order],
                color=color,
                alpha=0.65,
                linewidth=0.8,
            )
        axes[0].plot(
            sorted_days,
            quantiles[1],
            color="black",
            linewidth=1.8,
            label="Mean across tiles",
        )
        axes[0].set_title(
            f"Per-tile ice trajectories ({tile_thickness_quantiles.shape[1]} tiles)"
        )
    else:
        for index in draw_indices:
            axes[0].plot(
                sorted_days,
                sorted_thickness[index],
                color="#999999",
                alpha=0.10,
                linewidth=0.7,
            )
        axes[0].fill_between(
            sorted_days,
            quantiles[0],
            quantiles[2],
            color="#56B4E9",
            alpha=0.35,
            label="16th-84th percentile",
        )
        axes[0].plot(
            sorted_days,
            quantiles[1],
            color="#0072B2",
            linewidth=2.0,
            label="Median",
        )
        axes[0].set_title("Shared ice trajectory")
    if tile_rate_trajectory or has_rate_trajectory:
        axes[0].set_xlabel("")
    else:
        axes[0].set_xlabel("Days since first exposure")
    axes[0].set_ylabel("Ice thickness (nm)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    if tile_rate_trajectory:
        tile_rate_quantiles = np.percentile(tile_rates, [16, 50, 84], axis=0)
        rate_colors = plt.get_cmap("viridis", tile_rate_quantiles.shape[1])
        for tile_index in range(tile_rate_quantiles.shape[1]):
            color = rate_colors(tile_index)
            axes[1].fill_between(
                sorted_days,
                tile_rate_quantiles[0, tile_index, order],
                tile_rate_quantiles[2, tile_index, order],
                color=color,
                alpha=0.025,
            )
            axes[1].plot(
                sorted_days,
                tile_rate_quantiles[1, tile_index, order],
                color=color,
                alpha=0.65,
                linewidth=0.8,
            )
        axes[1].plot(
            sorted_days,
            np.median(np.mean(tile_rates, axis=1)[:, order], axis=0),
            color="black",
            linewidth=1.8,
            label="Mean across tiles",
        )
        axes[1].set_title(
            f"Per-tile instantaneous rates ({tile_rate_quantiles.shape[1]} tiles)"
        )
    elif has_rate_trajectory:
        sorted_rates = instantaneous_rates[:, order]
        rate_quantiles = np.percentile(sorted_rates, [16, 50, 84], axis=0)
        for index in draw_indices:
            axes[1].plot(
                sorted_days,
                sorted_rates[index],
                color="#999999",
                alpha=0.10,
                linewidth=0.7,
            )
        axes[1].fill_between(
            sorted_days,
            rate_quantiles[0],
            rate_quantiles[2],
            color="#E69F00",
            alpha=0.35,
            label="16th-84th percentile",
        )
        axes[1].plot(
            sorted_days,
            rate_quantiles[1],
            color="#D55E00",
            linewidth=2.0,
            label="Median",
        )
        axes[1].set_xlabel("Days since first exposure")
        axes[1].set_ylabel("Instantaneous accumulation rate (nm/day)")
        axes[1].set_title("Instantaneous accumulation rate")
        axes[1].legend(frameon=False, fontsize=8)
    else:
        rates = np.asarray(samples["mean_rate_nm_day"], dtype=float)
        axes[1].hist(rates, bins=30, color="#56B4E9", alpha=0.9)
        axes[1].axvline(np.median(rates), color="#D55E00", linewidth=2.0)
        axes[1].set_xlabel("Mean ice rate (nm/day)")
        axes[1].set_ylabel("Posterior samples")
        axes[1].set_title("Mean accumulation rate")
    axes[1].grid(alpha=0.2)
    figure.suptitle(f"{sample_scope(samples)} NumPyro hierarchical TMM fit")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_np_node_plot(path, samples, days, repeats, time_annotations=None):
    if "np_node_multipliers" not in samples:
        return
    repeat_values = np.asarray(sorted(set(repeats.tolist())), dtype=int)
    node_days = np.asarray(
        [np.mean(days[repeats == repeat]) for repeat in repeat_values],
        dtype=float,
    )
    fully_np_trajectory = bool(npz_scalar(samples, "fully_np_trajectory", 0))
    if fully_np_trajectory:
        if "node_instantaneous_rate_nm_day" not in samples:
            raise ValueError(
                "Fully non-parametric output is missing node instantaneous rates"
            )
        node_values = np.asarray(
            samples["node_instantaneous_rate_nm_day"], dtype=float
        )
        node_value_quantiles = np.percentile(node_values, [16, 50, 84], axis=0)
        node_ylabel = "Instantaneous rate (nm/day)"
        node_title = "Fully non-parametric observation-epoch-node rate"
    else:
        multipliers = np.asarray(samples["np_node_multipliers"], dtype=float)
        node_value_quantiles = np.percentile(multipliers, [16, 50, 84], axis=0)
        node_ylabel = "NP instantaneous-rate multiplier"
        node_title = "Observation-epoch-node rate modulation"

    thickness_quantiles = None
    if "np_node_thickness" in samples:
        thickness = np.asarray(samples["np_node_thickness"], dtype=float)
        if thickness.ndim == 3:
            thickness = np.mean(thickness, axis=1)
        thickness_quantiles = np.percentile(thickness, [16, 50, 84], axis=0)
    reset_day = (
        None
        if time_annotations is None
        else time_annotations.physical_reset_day
    )
    if thickness_quantiles is None:
        node_days, node_value_quantiles = split_time_series_at_reset(
            node_days,
            node_value_quantiles,
            reset_day=reset_day,
        )
    else:
        node_days, node_value_quantiles, thickness_quantiles = (
            split_time_series_at_reset(
                node_days,
                node_value_quantiles,
                thickness_quantiles,
                reset_day=reset_day,
            )
        )

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].fill_between(
        node_days,
        node_value_quantiles[0],
        node_value_quantiles[2],
        color="#56B4E9",
        alpha=0.35,
        label="16th-84th percentile",
    )
    axes[0].plot(
        node_days,
        node_value_quantiles[1],
        color="#0072B2",
        linewidth=2.0,
        label="Median",
    )
    if fully_np_trajectory:
        prior_rate = float(
            npz_scalar(samples, "trajectory_rate_prior_mean_nm_day", np.nan)
        )
        if np.isfinite(prior_rate) and prior_rate > 0.0:
            axes[0].axhline(
                prior_rate,
                color="black",
                linewidth=0.8,
                linestyle="--",
                label="Prior center",
            )
    else:
        axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel(node_ylabel)
    axes[0].set_title(node_title)
    axes[0].grid(alpha=0.2)

    if thickness_quantiles is not None:
        axes[1].fill_between(
            node_days,
            thickness_quantiles[0],
            thickness_quantiles[2],
            color="#E69F00",
            alpha=0.35,
            label="16th-84th percentile",
        )
        axes[1].plot(
            node_days,
            thickness_quantiles[1],
            color="#D55E00",
            linewidth=2.0,
            label="Median",
        )
        axes[1].set_ylabel("Ice thickness (nm)")
        axes[1].set_title("Ice thickness at observation-epoch nodes")
    axes[1].set_xlabel("Days since first exposure")
    axes[1].grid(alpha=0.2)
    if time_annotations is not None:
        annotate_time_axis(axes[0], time_annotations)
        annotate_time_axis(axes[1], time_annotations)
    for axis in axes:
        handles, labels = non_annotation_legend_handles(axis, time_annotations)
        if handles:
            axis.legend(handles, labels, frameon=False, fontsize=8)
    figure.suptitle(
        f"{sample_scope(samples)} "
        + (
            "fully non-parametric ice trajectory"
            if fully_np_trajectory
            else "non-parametric rate modulation"
        ),
        y=0.98,
    )
    if time_annotations is not None:
        figure.legend(
            handles=event_legend_handles(time_annotations),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=8,
            frameon=False,
            fontsize=7,
        )
    figure.subplots_adjust(top=0.88, bottom=0.15, hspace=0.22)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_tile_posterior(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No tile posterior rows found in {path}")
    return rows


def save_tile_map_plot(path, rows):
    n_rows = max(int(row["tile_row"]) for row in rows) + 1
    n_cols = max(int(row["tile_col"]) for row in rows) + 1
    if "ar_thickness_median" in rows[0]:
        metrics = (
            ("AR thickness scale", "ar_thickness"),
            ("AR refractive index", "ar_n"),
            ("AR absorption index", "ar_k"),
        )
    else:
        layer_count = 0
        while f"ar_layer{layer_count + 1}_thickness_median" in rows[0]:
            layer_count += 1
        if layer_count == 0:
            raise ValueError("Tile posterior has no recognized AR parameter columns")
        metrics = tuple(
            (f"AR layer {layer + 1} {label}", f"ar_layer{layer + 1}_{suffix}")
            for layer in range(layer_count)
            for label, suffix in (
                ("thickness scale", "thickness"),
                ("refractive index", "n"),
                ("absorption index", "k"),
            )
        )
    medians = []
    widths = []
    for _, prefix in metrics:
        median = np.full((n_rows, n_cols), np.nan)
        width = np.full((n_rows, n_cols), np.nan)
        for row in rows:
            tile_row = int(row["tile_row"])
            tile_col = int(row["tile_col"])
            median[tile_row, tile_col] = float(row[f"{prefix}_median"])
            width[tile_row, tile_col] = float(row[f"{prefix}_p84"]) - float(
                row[f"{prefix}_p16"]
            )
        medians.append(median)
        widths.append(width)

    figure, axes = plt.subplots(
        2,
        len(metrics),
        figsize=(4.6 * len(metrics), 8),
        squeeze=False,
        constrained_layout=True,
    )
    for column, ((label, _), median, width) in enumerate(zip(metrics, medians, widths)):
        for axis, values, title, cmap in (
            (axes[0, column], median, label, "viridis"),
            (axes[1, column], width, "68% width", "magma"),
        ):
            image = axis.imshow(values, origin="lower", interpolation="none", cmap=cmap)
            axis.set_xlabel("Tile column")
            axis.set_ylabel("Tile row")
            axis.set_title(title)
            axis.set_xticks(range(n_cols))
            axis.set_yticks(range(n_rows))
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            axis.grid(color="white", alpha=0.25, linewidth=0.5)
    detector = rows[0].get("detector", "WFI01")
    figure.suptitle(
        f"{detector} tile AR posterior; {len(rows)} selected of {n_rows * n_cols} tiles"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_tile_map_plots(output_dir, rows):
    detectors = sorted({row.get("detector", "WFI01") for row in rows})
    if len(detectors) == 1:
        path = output_dir / "numpyro_tile_ar_maps.png"
        save_tile_map_plot(path, rows)
        return [path]
    paths = []
    for detector in detectors:
        path = output_dir / f"numpyro_tile_ar_maps_{detector}.png"
        save_tile_map_plot(
            path,
            [row for row in rows if row.get("detector", "WFI01") == detector],
        )
        paths.append(path)
    return paths


def save_tile_rate_map_plot(path, rows):
    n_rows = max(int(row["tile_row"]) for row in rows) + 1
    n_cols = max(int(row["tile_col"]) for row in rows) + 1
    metrics = (
        ("Initial instantaneous rate", "initial_rate_nm_day"),
        ("Final instantaneous rate", "final_rate_nm_day"),
        ("Mean instantaneous rate", "mean_rate_nm_day"),
        ("Log-rate offset", "log_rate_offset"),
    )
    figure, axes = plt.subplots(
        2,
        len(metrics),
        figsize=(4.6 * len(metrics), 8),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (label, prefix) in enumerate(metrics):
        median = np.full((n_rows, n_cols), np.nan)
        width = np.full((n_rows, n_cols), np.nan)
        for row in rows:
            tile_row = int(row["tile_row"])
            tile_col = int(row["tile_col"])
            median[tile_row, tile_col] = float(row[f"{prefix}_median"])
            width[tile_row, tile_col] = float(row[f"{prefix}_p84"]) - float(
                row[f"{prefix}_p16"]
            )
        for axis, values, title, cmap in (
            (axes[0, column], median, label, "viridis"),
            (axes[1, column], width, "68% width", "magma"),
        ):
            image = axis.imshow(values, origin="lower", interpolation="none", cmap=cmap)
            axis.set_xlabel("Tile column")
            axis.set_ylabel("Tile row")
            axis.set_title(title)
            axis.set_xticks(range(n_cols))
            axis.set_yticks(range(n_rows))
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            axis.grid(color="white", alpha=0.25, linewidth=0.5)
    detector = rows[0].get("detector", "WFI01")
    figure.suptitle(f"{detector} tile accumulation-rate field; {len(rows)} selected tiles")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_tile_rate_map_plots(output_dir, rows):
    detectors = sorted({row.get("detector", "WFI01") for row in rows})
    paths = []
    for detector in detectors:
        path = output_dir / f"numpyro_tile_rate_maps_{detector}.png"
        save_tile_rate_map_plot(
            path,
            [row for row in rows if row.get("detector", "WFI01") == detector],
        )
        paths.append(path)
    return paths


def _load_prediction_inputs(
    output_dir,
    tile_statistics,
    model_path,
    sed_mode,
    wavelength_stride,
    wavelength_step_nm,
    max_tiles=None,
):
    from fit_wfi_tmm_numpyro import (
        build_prediction_function,
        apply_fixed_corrections,
        load_response_context,
        load_tile_statistics,
        require_numpyro,
        statistics_fingerprint,
        statistics_row_fingerprints,
    )

    jax, jnp, _, _, _, _, _ = require_numpyro()
    jax.config.update("jax_enable_x64", True)
    with np.load(output_dir / "posterior_samples.npz") as samples:
        ice_thickness = np.asarray(samples["ice_thickness"], dtype=float).copy()
        ar_parameters = np.asarray(samples["ar_parameters"], dtype=float).copy()
        acm_step = np.asarray(samples["acm_step_log_offset"], dtype=float).copy()
        fit_post_reset_ice_index = bool(
            npz_scalar(samples, "fit_post_reset_ice_index", "post_reset_ice_index" in samples)
        )
        fit_pre_post_ice_index = bool(
            npz_scalar(samples, "fit_pre_post_ice_index", "pre_reset_ice_n" in samples)
        )
        archived_ice_reset_day = npz_scalar(samples, "ice_reset_day", np.nan)
        post_reset_ice_reset_day = (
            None if np.isnan(archived_ice_reset_day) else float(archived_ice_reset_day)
        )
        post_reset_ice_index = (
            np.asarray(samples["post_reset_ice_index"], dtype=float).copy()
            if fit_post_reset_ice_index
            else None
        )
        pre_reset_ice_index = (
            np.asarray(samples["pre_reset_ice_n"], dtype=float).copy()
            + 1j * np.asarray(samples["pre_reset_ice_k"], dtype=float).copy()
            if fit_pre_post_ice_index
            else None
        )
        if fit_pre_post_ice_index:
            post_reset_ice_index = (
                np.asarray(samples["post_reset_ice_n"], dtype=float).copy()
                + 1j * np.asarray(samples["post_reset_ice_k"], dtype=float).copy()
            )
        fit_incidence_angle = bool(
            npz_scalar(samples, "fit_incidence_angle", 0)
        )
        incidence_angle_deg = (
            np.asarray(samples["incidence_angle_deg"], dtype=float).copy()
            if fit_incidence_angle and "incidence_angle_deg" in samples
            else None
        )
        if fit_incidence_angle and incidence_angle_deg is None:
            raise ValueError("Fitted-incidence posterior has no angle samples")
        fit_hgcdte = bool(npz_scalar(samples, "fit_hgcdte", 0))
        fit_hgcdte_modulation = bool(
            npz_scalar(samples, "fit_hgcdte_modulation", 0)
        )
        fit_ice_optical_modulation = bool(
            npz_scalar(samples, "fit_ice_optical_modulation", 0)
        )
        fit_post_reset_tile_log_offset = bool(
            npz_scalar(
                samples,
                "fit_post_reset_tile_log_offset",
                "post_reset_tile_log_offset" in samples,
            )
        )
        post_reset_tile_log_offsets = (
            np.asarray(samples["post_reset_tile_log_offset"], dtype=float).copy()
            if fit_post_reset_tile_log_offset
            else None
        )
        absolute_substrate_power = bool(
            npz_scalar(samples, "absolute_substrate_power", 0)
        )
        shared_absolute_hgcdte_scales = bool(
            npz_scalar(samples, "shared_absolute_hgcdte_scales", 0)
        )
        absolute_hgcdte_tile_variation = False
        hgcdte_parameters = None
        if absolute_substrate_power:
            n_log_scale = np.asarray(
                samples["absolute_hgcdte_n_log_scale"], dtype=float
            )
            k_log_scale = np.asarray(
                samples["absolute_hgcdte_k_log_scale"], dtype=float
            )
            absolute_hgcdte_tile_variation = bool(
                npz_scalar(
                    samples,
                    "absolute_hgcdte_tile_variation",
                    int(n_log_scale.ndim >= 3),
                )
            )
            if absolute_hgcdte_tile_variation:
                hgcdte_parameters = np.stack(
                    (n_log_scale, k_log_scale), axis=-1
                ).reshape(n_log_scale.shape[0], -1, 2)
            elif shared_absolute_hgcdte_scales:
                if n_log_scale.ndim == 1:
                    n_log_scale = n_log_scale[:, None]
                    k_log_scale = k_log_scale[:, None]
                if n_log_scale.ndim != 2 or k_log_scale.shape != n_log_scale.shape:
                    raise ValueError("Shared HgCdTe posterior has an unexpected shape")
                hgcdte_parameters = np.stack(
                    (n_log_scale, k_log_scale), axis=-1
                )
            else:
                hgcdte_parameters = np.column_stack(
                    (n_log_scale, k_log_scale)
                )
        elif fit_hgcdte:
            hgcdte_parameters = np.column_stack(
                (
                    np.asarray(samples["hgcdte_n"], dtype=float),
                    np.asarray(samples["hgcdte_k"], dtype=float),
                )
            )
        elif fit_hgcdte_modulation:
            hgcdte_parameters = np.concatenate(
                (
                    np.asarray(samples["hgcdte_n_log_modulation"], dtype=float),
                    np.asarray(samples["hgcdte_k_log_modulation"], dtype=float),
                ),
                axis=1,
            )
        ice_modulation_parameters = None
        if fit_ice_optical_modulation:
            ice_modulation_parameters = np.concatenate(
                (
                    np.asarray(samples["ice_n_log_modulation"], dtype=float),
                    np.asarray(samples["ice_k_log_modulation"], dtype=float),
                ),
                axis=1,
            )
        fit_conductive_layer = bool(
            npz_scalar(samples, "fit_conductive_layer", 0)
        )
        finite_hgcdte_detector = bool(
            npz_scalar(samples, "finite_hgcdte_detector", 0)
        )
        detector_backside_index = float(
            npz_scalar(samples, "detector_backside_index", 1.0)
        )
        detector_thickness = (
            np.asarray(samples["detector_thickness_nm"], dtype=float).copy()
            if finite_hgcdte_detector and "detector_thickness_nm" in samples
            else None
        )
        shared_ar_parameters_across_tiles = bool(
            npz_scalar(samples, "shared_ar_parameters_across_tiles", 0)
        )
        conductive_parameters = None
        if fit_conductive_layer:
            conductive_parameters = np.column_stack(
                (
                    np.asarray(
                        samples["conductive_log_thickness_nm"], dtype=float
                    ),
                    np.asarray(
                        samples["conductive_log_plasma_strength"], dtype=float
                    ),
                    np.asarray(
                        samples["conductive_log_damping_ratio"], dtype=float
                    ),
                )
            )
        fit_qe_rbf = bool(npz_scalar(samples, "fit_qe_rbf", 0))
        use_asw_optical_constants = bool(
            npz_scalar(samples, "use_asw_optical_constants", 0)
        )
        ice_optical_constants_source = (
            str(
                npz_scalar(
                    samples,
                    "ice_optical_constants_source",
                    ASW_OPTICAL_CONSTANTS_URL,
                )
            )
            if use_asw_optical_constants
            else None
        )
        sed_weighting = str(npz_scalar(samples, "sed_weighting", "photon"))
        use_hgcdte_optical_constants = bool(
            npz_scalar(samples, "use_hgcdte_optical_constants", 0)
        )
        incidence_cone_half_angle_deg = float(
            npz_scalar(samples, "incidence_cone_half_angle_deg", 0.0)
        )
        incidence_pupil_half_angle_deg = float(
            npz_scalar(samples, "incidence_pupil_half_angle_deg", 0.0)
        )
        incidence_chief_ray_tilts_deg = (
            np.asarray(samples["incidence_chief_ray_tilts_deg"], dtype=float)
            if "incidence_chief_ray_tilts_deg" in samples
            else None
        )
        qe_rbf_node_wavelengths_nm = (
            np.asarray(
                samples["qe_rbf_node_wavelengths_nm"], dtype=float
            ).reshape(-1)
            if "qe_rbf_node_wavelengths_nm" in samples
            else QE_RBF_NODE_WAVELENGTHS_NM.copy()
        )
        qe_rbf_width_nm = float(
            npz_scalar(samples, "qe_rbf_width_nm", QE_RBF_WIDTH_NM)
        )
        qe_rbf_parameters = None
        if fit_qe_rbf:
            qe_rbf_parameters = np.asarray(
                samples["qe_rbf_log_modulation"], dtype=float
            ).copy()
        posterior_sca_names = (
            np.asarray(samples["sca_names"]).astype(str)
            if "sca_names" in samples
            else None
        )
        posterior_tile_ids = (
            np.asarray(samples["tile_ids"], dtype=int)
            if "tile_ids" in samples
            else None
        )
        posterior_statistics_fingerprint = (
            str(np.asarray(samples["statistics_fingerprint"]).reshape(-1)[0])
            if "statistics_fingerprint" in samples
            else None
        )
        posterior_row_fingerprints = (
            np.asarray(samples["statistics_row_fingerprints"]).astype(str)
            if "statistics_row_fingerprints" in samples
            else None
        )
        archived_min_observation_day = npz_scalar(
            samples, "min_observation_day", np.nan
        )
        min_observation_day = (
            None
            if np.isnan(archived_min_observation_day)
            else float(archived_min_observation_day)
        )
        archived_early_time_cutoff_day = npz_scalar(
            samples, "early_time_scale_cutoff_day", np.nan
        )
        early_time_cutoff_day = (
            None
            if np.isnan(archived_early_time_cutoff_day)
            else float(archived_early_time_cutoff_day)
        )
        early_time_scale = (
            np.asarray(samples["early_time_scale"], dtype=float).copy()
            if early_time_cutoff_day is not None and "early_time_scale" in samples
            else None
        )
        fit_late_acm_step = bool(npz_scalar(samples, "fit_late_acm_step", 0))
        late_acm_step = (
            np.asarray(samples["late_acm_step_log_offset"], dtype=float).copy()
            if fit_late_acm_step and "late_acm_step_log_offset" in samples
            else None
        )
        fit_additional_acm_step = bool(
            npz_scalar(samples, "fit_additional_acm_step", 0)
        )
        additional_acm_step = (
            np.asarray(
                samples["additional_acm_step_log_offset"], dtype=float
            ).copy()
            if fit_additional_acm_step
            and "additional_acm_step_log_offset" in samples
            else None
        )
        archived_additional_acm_step_repeat = int(
            npz_scalar(samples, "additional_acm_step_repeat", -1)
        )
        archived_pre_reset_acm_step_repeat = int(
            npz_scalar(samples, "pre_reset_acm_step_repeat", -1)
        )
        fixed_corrections_path = str(
            npz_scalar(samples, "fixed_corrections_path", "")
        )
        fixed_corrections_applied = bool(
            npz_scalar(samples, "fixed_corrections_applied", 0)
        )
    data = load_tile_statistics(
        tile_statistics,
        max_tiles=max_tiles,
        max_groups=ice_thickness.shape[-1],
        min_observation_day=min_observation_day,
        additional_acm_step_repeat=(
            archived_additional_acm_step_repeat
            if archived_additional_acm_step_repeat >= 0
            else None
        ),
        pre_reset_acm_step_repeat=(
            archived_pre_reset_acm_step_repeat
            if archived_pre_reset_acm_step_repeat >= 0
            else None
        ),
    )
    absolute_hgcdte_parameters_rowwise = absolute_hgcdte_tile_variation
    if shared_absolute_hgcdte_scales:
        if hgcdte_parameters is None or hgcdte_parameters.ndim != 3:
            raise ValueError("Shared HgCdTe posterior does not have SCA parameters")
        hgcdte_parameters = hgcdte_parameters[
            :, np.asarray(data["sca_index"], dtype=int), :
        ]
        absolute_hgcdte_parameters_rowwise = True
    if fixed_corrections_applied:
        if not fixed_corrections_path:
            raise ValueError(
                "Posterior says fixed corrections were applied but no correction path was archived"
            )
        correction_path = resolve_archived_path(fixed_corrections_path)
        if correction_path is None or not correction_path.exists():
            raise FileNotFoundError(
                f"Archived fixed-correction file does not exist: {fixed_corrections_path}"
            )
        data = apply_fixed_corrections(data, correction_path)
    if posterior_sca_names is not None and not np.array_equal(
        data["sca_names"].astype(str), posterior_sca_names
    ):
        raise ValueError("Tile statistics SCAs do not match the archived posterior")
    if posterior_tile_ids is not None and not np.all(
        np.isin(data["tile_ids"], posterior_tile_ids)
    ):
        raise ValueError("Tile statistics tiles do not match the archived posterior")
    if posterior_statistics_fingerprint is not None:
        data_fingerprint = statistics_fingerprint(data)
        if data_fingerprint != posterior_statistics_fingerprint:
            if (
                posterior_row_fingerprints is None
                or posterior_sca_names is None
                or posterior_tile_ids is None
            ):
                raise ValueError("Tile statistics values do not match the archived posterior")
            posterior_tile_ids = np.asarray(posterior_tile_ids, dtype=int)
            posterior_row_count = len(posterior_sca_names) * len(posterior_tile_ids)
            if len(posterior_row_fingerprints) != posterior_row_count:
                raise ValueError("Archived tile fingerprints do not match the posterior")
            posterior_sca_indices = {
                name: index for index, name in enumerate(posterior_sca_names)
            }
            posterior_tile_indices = {
                int(tile_id): index
                for index, tile_id in enumerate(posterior_tile_ids)
            }
            selected_indices = []
            for sca_name in data["sca_names"].astype(str):
                if sca_name not in posterior_sca_indices:
                    raise ValueError("Tile statistics SCAs do not match the archived posterior")
                for tile_id in data["tile_ids"]:
                    if int(tile_id) not in posterior_tile_indices:
                        raise ValueError("Tile statistics tiles do not match the archived posterior")
                    selected_indices.append(
                        posterior_sca_indices[sca_name] * len(posterior_tile_ids)
                        + posterior_tile_indices[int(tile_id)]
                    )
            expected_fingerprints = posterior_row_fingerprints[selected_indices]
            if not np.array_equal(
                statistics_row_fingerprints(data), expected_fingerprints
            ):
                raise ValueError("Tile statistics values do not match the archived posterior")
    with np.load(output_dir / "posterior_samples.npz") as samples:
        for data_key, sample_key in (
            ("days", "observation_days"),
            ("led", "observation_led"),
            ("repeat", "observation_repeat"),
            ("exposure_times", "observation_exposure_times"),
            ("step_design", "observation_step_design"),
            ("late_step_design", "observation_late_step_design"),
            ("additional_step_design", "observation_additional_step_design"),
            ("pre_reset_step_design", "observation_pre_reset_step_design"),
        ):
            if sample_key in samples and not np.array_equal(
                data[data_key], np.asarray(samples[sample_key])
            ):
                raise ValueError("Tile statistics observations do not match the archived posterior")
    context = load_response_context(
        model_path,
        sed_mode,
        wavelength_stride,
        data["sca_names"],
        wavelength_step_nm,
        ice_optical_constants_source,
        use_hgcdte_optical_constants
        or fit_hgcdte_modulation
        or absolute_substrate_power,
        sed_weighting=sed_weighting,
        absolute_substrate_power=absolute_substrate_power,
    )
    context["qe_rbf_node_wavelengths_nm"] = qe_rbf_node_wavelengths_nm
    context["qe_rbf_width_nm"] = qe_rbf_width_nm
    context["use_asw_optical_constants"] = use_asw_optical_constants
    context["use_hgcdte_optical_constants"] = (
        use_hgcdte_optical_constants
        or fit_hgcdte_modulation
        or absolute_substrate_power
    )
    context["fit_hgcdte_modulation"] = fit_hgcdte_modulation
    context["fit_ice_optical_modulation"] = fit_ice_optical_modulation
    context["absolute_substrate_power"] = absolute_substrate_power
    context["absolute_hgcdte_tile_variation"] = absolute_hgcdte_tile_variation
    context["absolute_hgcdte_parameters_rowwise"] = absolute_hgcdte_parameters_rowwise
    context["shared_ar_parameters_across_tiles"] = (
        shared_ar_parameters_across_tiles
    )
    context["incidence_cone_half_angle_deg"] = incidence_cone_half_angle_deg
    context["incidence_pupil_half_angle_deg"] = incidence_pupil_half_angle_deg
    if ice_thickness.ndim not in (2, 3) or ice_thickness.shape[-1] != data["counts"].shape[1]:
        raise ValueError("Posterior groups do not match the tile statistics")
    if ar_parameters.ndim == 4:
        ar_parameters = ar_parameters.reshape(
            ar_parameters.shape[0], -1, ar_parameters.shape[-1]
        )
    if ar_parameters.ndim != 3:
        raise ValueError("Posterior tiles do not match the tile statistics")
    if ar_parameters.shape[1] != data["counts"].shape[0]:
        if max_tiles is None or ar_parameters.shape[1] % len(data["sca_names"]):
            raise ValueError("Posterior tiles do not match the tile statistics")
        full_data = load_tile_statistics(
            tile_statistics, min_observation_day=min_observation_day
        )
        posterior_tiles = ar_parameters.shape[1] // len(data["sca_names"])
        if (
            posterior_tiles != len(full_data["tile_ids"])
            or len(full_data["sca_names"]) != len(data["sca_names"])
            or not np.array_equal(full_data["sca_names"], data["sca_names"])
        ):
            raise ValueError("Posterior tiles do not match the tile statistics")
        selected_indices = np.asarray(
            [
                np.flatnonzero(full_data["tile_ids"] == tile_id)[0]
                for tile_id in data["tile_ids"]
            ],
            dtype=int,
        )
        selected_row_indices = np.concatenate(
            [
                sca_index * posterior_tiles + selected_indices
                for sca_index in range(len(data["sca_names"]))
            ]
        )
        if ice_thickness.ndim == 3:
            ice_thickness = ice_thickness[:, selected_row_indices, :]
        if detector_thickness is not None:
            detector_thickness = detector_thickness.reshape(
                detector_thickness.shape[0], len(data["sca_names"]), posterior_tiles
            )
            detector_thickness = detector_thickness[:, :, selected_indices].reshape(
                detector_thickness.shape[0], -1
            )
        if incidence_chief_ray_tilts_deg is not None:
            incidence_chief_ray_tilts_deg = incidence_chief_ray_tilts_deg[
                selected_row_indices
            ]
        ar_parameters = ar_parameters.reshape(
            ar_parameters.shape[0], len(data["sca_names"]), posterior_tiles, -1
        )
        ar_parameters = ar_parameters[:, :, selected_indices, :].reshape(
            ar_parameters.shape[0], -1, ar_parameters.shape[-1]
        )
        if post_reset_tile_log_offsets is not None:
            post_reset_tile_log_offsets = post_reset_tile_log_offsets.reshape(
                post_reset_tile_log_offsets.shape[0],
                len(data["sca_names"]),
                posterior_tiles,
            )
            post_reset_tile_log_offsets = post_reset_tile_log_offsets[
                :, :, selected_indices
            ].reshape(post_reset_tile_log_offsets.shape[0], -1)
        if (
            absolute_hgcdte_tile_variation
            and hgcdte_parameters is not None
        ):
            hgcdte_parameters = hgcdte_parameters.reshape(
                hgcdte_parameters.shape[0],
                len(data["sca_names"]),
                posterior_tiles,
                2,
            )
            hgcdte_parameters = hgcdte_parameters[:, :, selected_indices, :].reshape(
                hgcdte_parameters.shape[0], -1, 2
            )
    if ar_parameters.shape[1] != data["counts"].shape[0]:
        raise ValueError("Posterior tiles do not match the tile statistics")
    if post_reset_tile_log_offsets is not None:
        if post_reset_tile_log_offsets.ndim == 3:
            post_reset_tile_log_offsets = post_reset_tile_log_offsets.reshape(
                post_reset_tile_log_offsets.shape[0], -1
            )
        elif post_reset_tile_log_offsets.ndim != 2:
            raise ValueError("Posterior post-reset tile offsets have an unexpected shape")
        if post_reset_tile_log_offsets.shape[1] != data["counts"].shape[0]:
            raise ValueError("Posterior post-reset tile offsets do not match the tile statistics")
    if absolute_hgcdte_tile_variation:
        if (
            hgcdte_parameters is None
            or hgcdte_parameters.ndim != 3
            or hgcdte_parameters.shape[1] != data["counts"].shape[0]
        ):
            raise ValueError("Posterior HgCdTe tiles do not match the tile statistics")
    if absolute_hgcdte_parameters_rowwise:
        if (
            hgcdte_parameters is None
            or hgcdte_parameters.ndim != 3
            or hgcdte_parameters.shape[1] != data["counts"].shape[0]
        ):
            raise ValueError("Posterior HgCdTe rows do not match the tile statistics")
    if detector_thickness is not None:
        detector_thickness = detector_thickness.reshape(
            detector_thickness.shape[0], -1
        )
        if detector_thickness.shape[1] != data["counts"].shape[0]:
            raise ValueError("Posterior detector thickness does not match the tile statistics")
    if fit_post_reset_tile_log_offset:
        if post_reset_tile_log_offsets is None:
            raise ValueError("Post-reset tile offset posterior is missing")
        if post_reset_ice_reset_day is None:
            raise ValueError("Post-reset tile offsets require an archived reset day")
        median_offsets = np.median(post_reset_tile_log_offsets, axis=0)
        offset_mask = data["days"] >= post_reset_ice_reset_day
        correction = median_offsets[:, None] * offset_mask[None, :]
        data = dict(data)
        counts = np.asarray(data["counts"], dtype=float)
        sums = np.asarray(data["sums"], dtype=float)
        data["sums"] = sums + counts * correction
        data["sum_squared"] = np.asarray(data["sum_squared"], dtype=float) + (
            2.0 * correction * sums + counts * correction * correction
        )
        data["post_reset_tile_log_offset_applied"] = median_offsets
    context["finite_hgcdte_detector"] = finite_hgcdte_detector
    context["detector_backside_index"] = detector_backside_index
    if incidence_chief_ray_tilts_deg is None:
        incidence_chief_ray_tilts_deg = np.zeros((data["counts"].shape[0], 2))
    if incidence_chief_ray_tilts_deg.shape != (data["counts"].shape[0], 2):
        raise ValueError("Archived chief-ray geometry does not match the selected tiles")
    context["incidence_chief_ray_tilts_deg"] = incidence_chief_ray_tilts_deg
    context["fit_incidence_angle"] = fit_incidence_angle
    context["incidence_angle_deg"] = incidence_angle_deg
    context["fit_post_reset_ice_index"] = fit_post_reset_ice_index
    context["fit_pre_post_ice_index"] = fit_pre_post_ice_index
    context["post_reset_ice_reset_day"] = post_reset_ice_reset_day
    context["post_reset_ice_index"] = post_reset_ice_index
    context["pre_reset_ice_index"] = pre_reset_ice_index

    return (
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
    )


def apply_early_time_data_correction(
    data, prediction_values, early_time_scale, early_time_cutoff_day
):
    """Apply the posterior-median count correction to observations for plotting."""
    if early_time_cutoff_day is None:
        return data, prediction_values
    if early_time_scale is None:
        raise ValueError("Early-time scale is required when the cutoff is active")
    scale_samples = np.asarray(early_time_scale, dtype=float)
    if scale_samples.ndim == 1:
        scale = scale_samples
    elif scale_samples.ndim == 2:
        scale = np.median(scale_samples, axis=0)
    else:
        raise ValueError("Early-time scale posterior must be one- or two-dimensional")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Early-time scale posterior must be finite and positive")
    configuration = early_time_scale_configuration(data, early_time_cutoff_day)
    if scale.size != len(configuration["days"]):
        raise ValueError("Early-time scale posterior does not match the data nodes")
    observation_scale_index = np.asarray(
        configuration["observation_scale_index"], dtype=int
    )
    safe_scale_index = np.maximum(observation_scale_index, 0)
    log_correction = np.where(
        observation_scale_index >= 0,
        np.log(scale[safe_scale_index]),
        0.0,
    )

    corrected_data = dict(data)
    counts = np.asarray(data["counts"], dtype=float)
    sums = np.asarray(data["sums"], dtype=float)
    correction = log_correction[None, :]
    corrected_data["sums"] = sums + counts * correction
    if "sum_squared" in data:
        sum_squared = np.asarray(data["sum_squared"], dtype=float)
        corrected_data["sum_squared"] = (
            sum_squared + 2.0 * correction * sums + counts * correction * correction
        )
    corrected_data["early_time_scale_applied_to_data"] = True
    corrected_data["early_time_scale_log_correction"] = log_correction
    # Predictions were evaluated with the early-time model offset disabled.
    return corrected_data, np.asarray(prediction_values, dtype=float)


def load_model_predictions(
    output_dir,
    tile_statistics,
    model_path,
    sed_mode,
    wavelength_stride,
    wavelength_step_nm,
    max_tiles=None,
):
    """Load all posterior predictions, evaluating the TMM in row chunks."""
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
    ) = _load_prediction_inputs(
        output_dir,
        tile_statistics,
        model_path,
        sed_mode,
        wavelength_stride,
        wavelength_step_nm,
        max_tiles,
    )
    fit_hgcdte_modulation = bool(context["fit_hgcdte_modulation"])
    fit_ice_optical_modulation = bool(context["fit_ice_optical_modulation"])
    absolute_substrate_power = bool(context["absolute_substrate_power"])
    fit_incidence_angle = bool(context["fit_incidence_angle"])
    fit_ice_index = bool(
        context["fit_post_reset_ice_index"] or context["fit_pre_post_ice_index"]
    )
    prediction_chunks = []
    for start in range(0, len(ice_thickness), 16):
        stop = min(start + 16, len(ice_thickness))
        row_chunks = []
        for row_start in range(
            0, ar_parameters.shape[1], PREDICTION_ROW_CHUNK_SIZE
        ):
            row_stop = min(
                row_start + PREDICTION_ROW_CHUNK_SIZE, ar_parameters.shape[1]
            )
            row_indices = np.arange(row_start, row_stop, dtype=int)
            predictor = build_prediction_function(
                data,
                context,
                row_indices=row_indices,
                fit_hgcdte=fit_hgcdte,
                fit_hgcdte_modulation=fit_hgcdte_modulation,
                fit_conductive_layer=fit_conductive_layer,
                finite_hgcdte_detector=context["finite_hgcdte_detector"],
                detector_backside_index=context["detector_backside_index"],
                fit_qe_rbf=fit_qe_rbf,
                qe_rbf_node_wavelengths_nm=context["qe_rbf_node_wavelengths_nm"],
                qe_rbf_width_nm=context["qe_rbf_width_nm"],
                use_asw_optical_constants=context["use_asw_optical_constants"],
                fit_ice_optical_modulation=fit_ice_optical_modulation,
                use_hgcdte_optical_constants=context["use_hgcdte_optical_constants"],
                incidence_cone_half_angle_deg=context["incidence_cone_half_angle_deg"],
                incidence_pupil_half_angle_deg=context["incidence_pupil_half_angle_deg"],
                incidence_chief_ray_tilts_deg=context["incidence_chief_ray_tilts_deg"],
                 fit_incidence_angle=fit_incidence_angle,
                 # Plot the physical model against data corrected below.
                 early_time_cutoff_day=None,
                 share_ar_across_rows=context["shared_ar_parameters_across_tiles"],
                 fit_late_acm_step=late_acm_step is not None,
                 fit_additional_acm_step=additional_acm_step is not None,
                 post_reset_ice_reset_day=(
                     context["post_reset_ice_reset_day"] if fit_ice_index else None
                 ),
             )
            predictor_in_axes = [0, 0, 0]
            predictor_in_axes.append(
                0
                if fit_hgcdte or fit_hgcdte_modulation or absolute_substrate_power
                else None
            )
            predictor_in_axes.append(0 if fit_conductive_layer else None)
            predictor_in_axes.append(0 if fit_qe_rbf else None)
            if late_acm_step is not None:
                predictor_in_axes.extend((None, 0))
            elif (
                detector_thickness is not None
                or fit_incidence_angle
                or additional_acm_step is not None
                or fit_ice_index
            ):
                predictor_in_axes.extend((None, None))
            if detector_thickness is not None:
                predictor_in_axes.append(0)
            elif fit_incidence_angle:
                predictor_in_axes.append(None)
            if fit_incidence_angle:
                predictor_in_axes.append(0)
            if additional_acm_step is not None:
                predictor_in_axes.append(0)
            elif fit_ice_index or fit_ice_optical_modulation:
                predictor_in_axes.append(None)
            if fit_ice_index:
                predictor_in_axes.extend(
                    (
                        0
                        if (
                            context["fit_post_reset_ice_index"]
                            or context["fit_pre_post_ice_index"]
                        )
                        else None,
                        0 if context["fit_pre_post_ice_index"] else None,
                    )
                )
            elif fit_ice_optical_modulation:
                predictor_in_axes.extend((None, None))
            if fit_ice_optical_modulation:
                predictor_in_axes.append(0)
            row_batched_predictor = jax.jit(
                jax.vmap(predictor, in_axes=tuple(predictor_in_axes))
            )
            prediction_args = (
                jnp.asarray(
                    ice_thickness[start:stop]
                    if ice_thickness.ndim == 2
                    else ice_thickness[start:stop, row_indices, :]
                ),
                jnp.asarray(ar_parameters[start:stop, row_indices]),
                jnp.asarray(acm_step[start:stop]),
            )
            prediction_args += (
                jnp.asarray(
                    hgcdte_parameters[start:stop, row_indices]
                    if context["absolute_hgcdte_parameters_rowwise"]
                    else hgcdte_parameters[start:stop]
                )
                if fit_hgcdte or fit_hgcdte_modulation or absolute_substrate_power
                else None,
            )
            prediction_args += (
                jnp.asarray(conductive_parameters[start:stop])
                if fit_conductive_layer
                else None,
            )
            prediction_args += (
                jnp.asarray(qe_rbf_parameters[start:stop])
                if fit_qe_rbf
                else None,
            )
            if late_acm_step is not None:
                prediction_args += (None, jnp.asarray(late_acm_step[start:stop]))
            elif (
                detector_thickness is not None
                or fit_incidence_angle
                or additional_acm_step is not None
                or fit_ice_index
            ):
                prediction_args += (None, None)
            if detector_thickness is not None:
                prediction_args += (
                    jnp.asarray(detector_thickness[start:stop, row_indices]),
                )
            elif fit_incidence_angle:
                prediction_args += (None,)
            if fit_incidence_angle:
                prediction_args += (
                    jnp.asarray(context["incidence_angle_deg"][start:stop]),
                )
            if additional_acm_step is not None:
                prediction_args += (
                    jnp.asarray(additional_acm_step[start:stop]),
                )
            elif fit_ice_index or fit_ice_optical_modulation:
                prediction_args += (None,)
            if fit_ice_index:
                prediction_args += (
                    jnp.asarray(context["post_reset_ice_index"][start:stop])
                    if (
                        context["fit_post_reset_ice_index"]
                        or context["fit_pre_post_ice_index"]
                    )
                    else None,
                    jnp.asarray(context["pre_reset_ice_index"][start:stop])
                    if context["fit_pre_post_ice_index"]
                    else None,
                )
            elif fit_ice_optical_modulation:
                prediction_args += (None, None)
            if fit_ice_optical_modulation:
                prediction_args += (
                    jnp.asarray(ice_modulation_parameters[start:stop]),
                )
            row_chunks.append(np.asarray(row_batched_predictor(*prediction_args)))
        prediction_chunks.append(np.concatenate(row_chunks, axis=1))
    return apply_early_time_data_correction(
        data,
        np.concatenate(prediction_chunks, axis=0),
        early_time_scale,
        early_time_cutoff_day,
    )


def load_model_prediction_quantiles(
    output_dir,
    tile_statistics,
    model_path,
    sed_mode,
    wavelength_stride,
    wavelength_step_nm,
    max_tiles=None,
):
    """Return 16th, 50th, and 84th percentile predictions without a full draw tensor."""
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
    ) = _load_prediction_inputs(
        output_dir,
        tile_statistics,
        model_path,
        sed_mode,
        wavelength_stride,
        wavelength_step_nm,
        max_tiles,
    )
    fit_hgcdte_modulation = bool(context["fit_hgcdte_modulation"])
    fit_ice_optical_modulation = bool(context["fit_ice_optical_modulation"])
    absolute_substrate_power = bool(context["absolute_substrate_power"])
    fit_incidence_angle = bool(context["fit_incidence_angle"])
    fit_ice_index = bool(
        context["fit_post_reset_ice_index"] or context["fit_pre_post_ice_index"]
    )
    quantile_chunks = []
    for row_start in range(
        0, ar_parameters.shape[1], PREDICTION_ROW_CHUNK_SIZE
    ):
        row_stop = min(
            row_start + PREDICTION_ROW_CHUNK_SIZE, ar_parameters.shape[1]
        )
        row_indices = np.arange(row_start, row_stop, dtype=int)
        predictor = build_prediction_function(
            data,
            context,
            row_indices=row_indices,
            fit_hgcdte=fit_hgcdte,
            fit_hgcdte_modulation=fit_hgcdte_modulation,
            fit_conductive_layer=fit_conductive_layer,
            finite_hgcdte_detector=context["finite_hgcdte_detector"],
            detector_backside_index=context["detector_backside_index"],
            fit_qe_rbf=fit_qe_rbf,
            qe_rbf_node_wavelengths_nm=context["qe_rbf_node_wavelengths_nm"],
            qe_rbf_width_nm=context["qe_rbf_width_nm"],
            use_asw_optical_constants=context["use_asw_optical_constants"],
            fit_ice_optical_modulation=fit_ice_optical_modulation,
            use_hgcdte_optical_constants=context["use_hgcdte_optical_constants"],
            incidence_cone_half_angle_deg=context["incidence_cone_half_angle_deg"],
            incidence_pupil_half_angle_deg=context["incidence_pupil_half_angle_deg"],
            incidence_chief_ray_tilts_deg=context["incidence_chief_ray_tilts_deg"],
            fit_incidence_angle=fit_incidence_angle,
            # Plot the physical model against data corrected below.
            early_time_cutoff_day=None,
            share_ar_across_rows=context["shared_ar_parameters_across_tiles"],
             fit_late_acm_step=late_acm_step is not None,
             fit_additional_acm_step=additional_acm_step is not None,
             absolute_substrate_power=absolute_substrate_power,
             post_reset_ice_reset_day=(
                 context["post_reset_ice_reset_day"] if fit_ice_index else None
             ),
         )
        predictor_in_axes = [0, 0, 0]
        predictor_in_axes.append(
            0
            if fit_hgcdte or fit_hgcdte_modulation or absolute_substrate_power
            else None
        )
        predictor_in_axes.append(0 if fit_conductive_layer else None)
        predictor_in_axes.append(0 if fit_qe_rbf else None)
        if late_acm_step is not None:
            predictor_in_axes.extend((None, 0))
        elif (
            detector_thickness is not None
            or fit_incidence_angle
            or additional_acm_step is not None
            or fit_ice_index
        ):
            predictor_in_axes.extend((None, None))
        if detector_thickness is not None:
            predictor_in_axes.append(0)
        elif fit_incidence_angle:
            predictor_in_axes.append(None)
        if fit_incidence_angle:
            predictor_in_axes.append(0)
        if additional_acm_step is not None:
            predictor_in_axes.append(0)
        elif fit_ice_index or fit_ice_optical_modulation:
            predictor_in_axes.append(None)
        if fit_ice_index:
            predictor_in_axes.extend(
                (
                    0
                    if (
                        context["fit_post_reset_ice_index"]
                        or context["fit_pre_post_ice_index"]
                    )
                    else None,
                    0 if context["fit_pre_post_ice_index"] else None,
                )
            )
        elif fit_ice_optical_modulation:
            predictor_in_axes.extend((None, None))
        if fit_ice_optical_modulation:
            predictor_in_axes.append(0)
        row_batched_predictor = jax.jit(
            jax.vmap(predictor, in_axes=tuple(predictor_in_axes))
        )
        sample_predictions = []
        for start in range(0, len(ice_thickness), 16):
            stop = min(start + 16, len(ice_thickness))
            prediction_args = (
                jnp.asarray(
                    ice_thickness[start:stop]
                    if ice_thickness.ndim == 2
                    else ice_thickness[start:stop, row_indices, :]
                ),
                jnp.asarray(ar_parameters[start:stop, row_indices]),
                jnp.asarray(acm_step[start:stop]),
            )
            prediction_args += (
                jnp.asarray(
                    hgcdte_parameters[start:stop, row_indices]
                    if context["absolute_hgcdte_parameters_rowwise"]
                    else hgcdte_parameters[start:stop]
                )
                if fit_hgcdte or fit_hgcdte_modulation or absolute_substrate_power
                else None,
            )
            prediction_args += (
                jnp.asarray(conductive_parameters[start:stop])
                if fit_conductive_layer
                else None,
            )
            prediction_args += (
                jnp.asarray(qe_rbf_parameters[start:stop])
                if fit_qe_rbf
                else None,
            )
            if late_acm_step is not None:
                prediction_args += (None, jnp.asarray(late_acm_step[start:stop]))
            elif (
                detector_thickness is not None
                or fit_incidence_angle
                or additional_acm_step is not None
                or fit_ice_index
            ):
                prediction_args += (None, None)
            if detector_thickness is not None:
                prediction_args += (
                    jnp.asarray(detector_thickness[start:stop, row_indices]),
                )
            elif fit_incidence_angle:
                prediction_args += (None,)
            if fit_incidence_angle:
                prediction_args += (
                    jnp.asarray(context["incidence_angle_deg"][start:stop]),
                )
            if additional_acm_step is not None:
                prediction_args += (
                    jnp.asarray(additional_acm_step[start:stop]),
                )
            elif fit_ice_index or fit_ice_optical_modulation:
                prediction_args += (None,)
            if fit_ice_index:
                prediction_args += (
                    jnp.asarray(context["post_reset_ice_index"][start:stop])
                    if (
                        context["fit_post_reset_ice_index"]
                        or context["fit_pre_post_ice_index"]
                    )
                    else None,
                    jnp.asarray(context["pre_reset_ice_index"][start:stop])
                    if context["fit_pre_post_ice_index"]
                    else None,
                )
            elif fit_ice_optical_modulation:
                prediction_args += (None, None)
            if fit_ice_optical_modulation:
                prediction_args += (jnp.asarray(ice_modulation_parameters),)
            sample_predictions.append(
                np.asarray(row_batched_predictor(*prediction_args))
            )
        row_predictions = np.concatenate(sample_predictions, axis=0)
        quantile_chunks.append(
            np.percentile(row_predictions, [16.0, 50.0, 84.0], axis=0)
        )
    return apply_early_time_data_correction(
        data,
        np.concatenate(quantile_chunks, axis=1),
        early_time_scale,
        early_time_cutoff_day,
    )


def save_tile_fit_plot(path, data, prediction_quantiles, time_annotations=None):
    observed = data["sums"] / data["counts"]
    days = data["days"]
    leds = data["led"]
    observation_label = (
        "Corrected tile observations"
        if data.get("early_time_scale_applied_to_data", False)
        else "Tile observations"
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True)
    reset_day = (
        None
        if time_annotations is None
        else time_annotations.physical_reset_day
    )
    for led, axis in enumerate(axes.flat, start=1):
        groups = np.flatnonzero(leds == led)
        order = groups[np.argsort(days[groups], kind="stable")]
        x = days[order]
        observed_led = observed[:, order]
        if observed_led.shape[0] == 1:
            predicted_curve_quantiles = prediction_quantiles[:, 0][:, order]
            model_band_label = "Posterior 16th-84th percentile"
        else:
            tile_medians = prediction_quantiles[1][:, order]
            predicted_curve_quantiles = np.asarray(
                [
                    np.percentile(tile_medians, 16.0, axis=0),
                    np.percentile(tile_medians, 50.0, axis=0),
                    np.percentile(tile_medians, 84.0, axis=0),
                ]
            )
            model_band_label = "Tile-median 16th-84th percentile"
        residual = observed_led - prediction_quantiles[1][:, order]
        x, observed_plot, predicted_plot = split_time_series_at_reset(
            x,
            observed_led,
            predicted_curve_quantiles,
            reset_day=reset_day,
        )
        axis.scatter(
            np.broadcast_to(x, observed_plot.shape).ravel(),
            (100.0 * np.expm1(observed_plot)).ravel(),
            color="#444444",
            alpha=0.18,
            s=7,
            label=observation_label if led == 1 else None,
        )
        axis.fill_between(
            x,
            100.0 * np.expm1(predicted_plot[0]),
            100.0 * np.expm1(predicted_plot[2]),
            color="#56B4E9",
            alpha=0.35,
            label=model_band_label if led == 1 else None,
        )
        axis.plot(
            x,
            100.0 * np.expm1(predicted_plot[1]),
            color="#0072B2",
            linewidth=2.0,
            label="Model median" if led == 1 else None,
        )
        rms = np.sqrt(np.mean(residual**2))
        axis.set_title(f"LED {led}; tile RMS={rms:.4g} log")
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Days since first exposure")
    if time_annotations is not None:
        for axis in axes.flat:
            annotate_time_axis(axis, time_annotations)
    axes[0, 0].set_ylabel("Relative change (%)")
    axes[1, 0].set_ylabel("Relative change (%)")
    handles, labels = non_annotation_legend_handles(
        axes[0, 0], time_annotations
    )
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, fontsize=8)
    figure.suptitle(
        f"{scope_label(data['sca_names'])} observations versus posterior physical TMM model",
        y=0.97,
    )
    if time_annotations is not None:
        figure.legend(
            handles=event_legend_handles(time_annotations),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=8,
            frameon=False,
            fontsize=7,
        )
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.89))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_qe_modulation_plot(path, samples, data):
    """Plot ground QE and the posterior wavelength-dependent modulation."""
    if "qe_rbf_log_modulation" not in samples:
        raise ValueError("QE RBF posterior samples are not present")
    node_wavelengths = (
        np.asarray(samples["qe_rbf_node_wavelengths_nm"], dtype=float).reshape(-1)
        if "qe_rbf_node_wavelengths_nm" in samples
        else QE_RBF_NODE_WAVELENGTHS_NM
    )
    width_nm = float(npz_scalar(samples, "qe_rbf_width_nm", QE_RBF_WIDTH_NM))
    qe_log_modulation = np.asarray(samples["qe_rbf_log_modulation"], dtype=float)
    qe_wavelengths, qe_values, _ = load_qe(QE_URL)
    basis = np.exp(
        -0.5
        * ((qe_wavelengths[:, None] - node_wavelengths[None, :]) / width_nm) ** 2
    )
    basis /= np.sum(basis, axis=1, keepdims=True)
    modulation = np.exp(qe_log_modulation @ basis.T)
    modulation_quantiles = np.percentile(modulation, [16.0, 50.0, 84.0], axis=0)

    figure, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )
    for sca_index, sca_name in enumerate(data["sca_names"]):
        qe = np.asarray(qe_values[:, data["qe_columns"][sca_index]], dtype=float)
        axes[0].plot(qe_wavelengths, qe, linewidth=1.2, label=f"{sca_name} ground QE")
        axes[0].plot(
            qe_wavelengths,
            qe * modulation_quantiles[1],
            linewidth=1.2,
            linestyle="--",
            label=f"{sca_name} modulated QE median",
        )
    axes[0].set_ylabel("Quantum efficiency")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_title("Ground QE and fitted wavelength-dependent modulation")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].grid(alpha=0.2)

    axes[1].fill_between(
        qe_wavelengths,
        modulation_quantiles[0],
        modulation_quantiles[2],
        color="#56B4E9",
        alpha=0.35,
        label="Posterior 16th-84th percentile",
    )
    axes[1].plot(
        qe_wavelengths,
        modulation_quantiles[1],
        color="#0072B2",
        linewidth=1.5,
        label="Posterior median multiplier",
    )
    axes[1].scatter(
        node_wavelengths,
        np.interp(node_wavelengths, qe_wavelengths, modulation_quantiles[1]),
        color="#D55E00",
        zorder=3,
        label="RBF nodes",
    )
    for wavelength in node_wavelengths:
        axes[1].axvline(wavelength, color="#D55E00", alpha=0.18, linewidth=0.8)
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("QE multiplier")
    axes[1].set_title(
        f"Normalized Gaussian RBF basis; width={width_nm:g} nm; "
        f"nodes={len(node_wavelengths)}"
    )
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_ice_modulation_plot(path, samples):
    """Plot posterior global n/k multipliers over the Mastrapa baseline."""
    from fit_wfi_tmm_numpyro import load_asw_optical_constants

    if "ice_n_log_modulation" not in samples or "ice_k_log_modulation" not in samples:
        raise ValueError("Ice optical modulation posterior samples are not present")
    source = str(npz_scalar(samples, "ice_optical_constants_source", ASW_OPTICAL_CONSTANTS_URL))
    wavelengths, _, _ = load_asw_optical_constants(source)
    node_wavelengths = np.asarray(
        samples["ice_modulation_node_wavelengths_nm"]
        if "ice_modulation_node_wavelengths_nm" in samples
        else np.asarray(
            [
                623.520885362797,
                852.802649193115,
                1038.5847494059,
                1269.7582849979,
                1481.24145201185,
                1670.16888078615,
            ]
        ),
        dtype=float,
    ).reshape(-1)
    width_nm = float(npz_scalar(samples, "ice_modulation_width_nm", 500.0))
    basis = np.exp(
        -0.5
        * ((wavelengths[:, None] - node_wavelengths[None, :]) / width_nm) ** 2
    )
    basis /= np.sum(basis, axis=1, keepdims=True)
    modulation_quantiles = []
    for name in ("ice_n_log_modulation", "ice_k_log_modulation"):
        log_modulation = np.asarray(samples[name], dtype=float)
        modulation_quantiles.append(
            np.percentile(np.exp(log_modulation @ basis.T), [16.0, 50.0, 84.0], axis=0)
        )

    figure, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, constrained_layout=True
    )
    for axis, quantiles, parameter in zip(
        axes, modulation_quantiles, ("n", "k")
    ):
        axis.fill_between(
            wavelengths,
            quantiles[0],
            quantiles[2],
            color="#56B4E9",
            alpha=0.35,
            label="Posterior 16th-84th percentile",
        )
        axis.plot(
            wavelengths,
            quantiles[1],
            color="#0072B2",
            linewidth=1.5,
            label="Posterior median multiplier",
        )
        axis.scatter(
            node_wavelengths,
            np.interp(node_wavelengths, wavelengths, quantiles[1]),
            color="#D55E00",
            zorder=3,
            label="Modulation nodes",
        )
        for wavelength in node_wavelengths:
            axis.axvline(wavelength, color="#D55E00", alpha=0.18, linewidth=0.8)
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_ylabel(f"{parameter}(lambda) multiplier")
        axis.set_title(f"Global ice {parameter}(lambda) modulation")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.2)
    axes[1].set_xlabel("Wavelength (nm)")
    figure.suptitle(
        f"Mastrapa baseline; normalized Gaussian width={width_nm:g} nm; "
        f"nodes={len(node_wavelengths)}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_tile_data_model_time_plot(
    path,
    data,
    prediction_quantiles,
    acm_step=0.0,
    late_acm_step=0.0,
    additional_acm_step=0.0,
    time_annotations=None,
):
    """Plot every LED after ACM handling; count correction is already in the data."""
    observed = np.asarray(data["sums"] / data["counts"], dtype=float)
    prediction_quantiles = np.asarray(prediction_quantiles, dtype=float).copy()
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    step_design = np.asarray(data["step_design"], dtype=float)
    late_step_design = np.asarray(
        data.get("late_step_design", np.zeros(len(days))), dtype=float
    )
    additional_step_design = np.asarray(
        data.get("additional_step_design", np.zeros(len(days))), dtype=float
    )
    acm_step = np.asarray(acm_step, dtype=float).reshape(-1)
    if acm_step.size == 1:
        group_acm_step = np.full(len(leds), acm_step[0])
    elif acm_step.size == 6:
        group_acm_step = acm_step[leds - 1]
    else:
        raise ValueError("ACM step must be scalar or have one value per LED")
    acm_correction = group_acm_step * step_design
    late_acm_step = float(np.asarray(late_acm_step).reshape(-1)[0])
    acm_correction += late_acm_step * late_step_design
    additional_acm_step = float(
        np.asarray(additional_acm_step).reshape(-1)[0]
    )
    acm_correction += additional_acm_step * additional_step_design
    observed -= acm_correction[None, :]
    prediction_quantiles -= acm_correction[None, None, :]

    figure, axis = plt.subplots(figsize=(13, 7))
    reset_day = (
        None
        if time_annotations is None
        else time_annotations.physical_reset_day
    )
    led_values = sorted(np.unique(leds))
    cmap = plt.get_cmap("tab10")
    color_by_led = {
        led: cmap(index / max(1, len(led_values) - 1))
        for index, led in enumerate(led_values)
    }
    for led in led_values:
        groups = np.flatnonzero(leds == led)
        order = groups[np.argsort(days[groups], kind="stable")]
        x = days[order]
        observed_led = observed[:, order]
        model_led = prediction_quantiles[1][:, order]
        x, observed_plot, model_plot = split_time_series_at_reset(
            x,
            observed_led,
            model_led,
            reset_day=reset_day,
        )
        for tile in range(observed_led.shape[0]):
            axis.scatter(
                x,
                100.0 * np.expm1(observed_plot[tile]),
                color=color_by_led[led],
                alpha=0.22,
                s=12,
                linewidths=0,
            )
        model_alpha = 0.22 if model_led.shape[0] > 1 else 1.0
        model_linewidth = 0.8 if model_led.shape[0] > 1 else 2.0
        for model_curve in model_plot:
            axis.plot(
                x,
                100.0 * np.expm1(model_curve),
                color=color_by_led[led],
                alpha=model_alpha,
                linewidth=model_linewidth,
            )

    color_handles = [
        Line2D(
            [],
            [],
            color=color_by_led[led],
            linewidth=2.0,
            label=f"LED {led}",
        )
        for led in led_values
    ]
    style_handles = [
        Line2D(
            [],
            [],
            color="#444444",
            marker="o",
            linestyle="",
            markersize=4,
            label="Data",
        ),
        Line2D(
            [],
            [],
            color="#444444",
            linewidth=2.0,
            label=(
                f"Individual tile models ({observed.shape[0]})"
                if observed.shape[0] > 1
                else "Model"
            ),
        ),
    ]
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.set_xlabel("Days since first exposure")
    if data.get("fixed_corrections_applied", False):
        axis.set_ylabel("Relative change after fixed gain/ACM correction (%)")
        title_suffix = "fixed gain/ACM corrections applied to data"
    else:
        axis.set_ylabel("Relative change after ACM-step removal (%)")
        title_suffix = "fitted ACM throughput step removed; count correction applied to data"
    axis.grid(alpha=0.2)
    event_handles = []
    if time_annotations is not None:
        annotate_time_axis(axis, time_annotations)
        event_handles = event_legend_handles(time_annotations)
    axis.legend(
        handles=color_handles + style_handles,
        frameon=False,
        fontsize=9,
        ncol=min(4, len(color_handles) + len(style_handles)),
    )
    figure.suptitle(
        f"{scope_label(data['sca_names'])} tile data and modeled curves; "
        + title_suffix,
        y=0.97,
    )
    if event_handles:
        figure.legend(
            handles=event_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=8,
            frameon=False,
            fontsize=7,
        )
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.89))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_corner_plot(path, values, labels, title):
    import corner

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Corner plot values must be 2D, got shape {values.shape}")
    labels = tuple(labels)
    if len(labels) != values.shape[1]:
        raise ValueError("Corner plot labels do not match the value columns")
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.size:
        spans = np.ptp(values, axis=0)
        scales = np.maximum(1.0, np.max(np.abs(values), axis=0))
        varying = spans > 1.0e-12 * scales
        values = values[:, varying]
        labels = tuple(label for label, keep in zip(labels, varying) if keep)
    if values.shape[1] == 0:
        raise ValueError("Corner plots require at least one varying finite column")
    if values.shape[0] <= values.shape[1]:
        raise ValueError(
            "Corner plots require more finite posterior samples than dimensions; "
            f"got {values.shape[0]} samples and {values.shape[1]} dimensions"
        )

    ranges = []
    for column in values.T:
        lower = float(np.min(column))
        upper = float(np.max(column))
        span = upper - lower
        padding = max(0.05 * span, 1.0e-6 * max(1.0, abs(lower), abs(upper)))
        ranges.append((lower - padding, upper + padding))

    figure = corner.corner(
        values,
        labels=labels,
        range=ranges,
        bins=30,
        color="#0072B2",
        fill_contours=True,
        plot_datapoints=False,
        smooth=0.8,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".4g",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
    )
    figure.suptitle(title, y=1.02, fontsize=14)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_global_corner_plots(output_dir, samples):
    ar_mean = np.asarray(samples["ar_mean"], dtype=float)
    ar_layers = ar_layer_count(samples)
    acm_step_samples, acm_step_labels = acm_step_corner_values(samples)
    global_values = np.column_stack(
        (
            samples["initial_thickness_nm"],
            samples["mean_rate_nm_day"],
            acm_step_samples,
            ar_mean,
        )
    )
    save_corner_plot(
        output_dir / "numpyro_global_corner.png",
        global_values,
        (
            "d0 (nm)",
            "mean rate (nm/day)",
        ) + acm_step_labels + ar_parameter_labels(ar_layers),
        f"{sample_scope(samples)} global posterior, marginalized over tile parameters",
    )

    if "ar_sigma_raw" in samples:
        sigma = np.asarray(samples["ar_sigma_raw"], dtype=float)
        hyper_values = np.column_stack(
            (ar_mean, sigma)
        )
        labels = ar_parameter_labels(ar_layers) + hierarchy_sigma_labels(
            ar_layers, sigma.shape[1], "tile"
        )
        title = f"{sample_scope(samples)} AR population hyperparameters"
    elif "ar_sca_sigma" in samples and "ar_tile_sigma" in samples:
        sca_sigma = np.asarray(samples["ar_sca_sigma"], dtype=float)
        tile_sigma = np.asarray(samples["ar_tile_sigma"], dtype=float)
        hyper_values = np.column_stack(
            (
                ar_mean,
                sca_sigma,
                tile_sigma,
            )
        )
        labels = (
            ar_parameter_labels(ar_layers)
            + hierarchy_sigma_labels(ar_layers, sca_sigma.shape[1], "SCA")
            + hierarchy_sigma_labels(ar_layers, tile_sigma.shape[1], "tile")
        )
        title = "All-SCA AR hierarchy hyperparameters"
    else:
        return None
    hyper_path = output_dir / "numpyro_ar_hyper_corner.png"
    save_corner_plot(
        hyper_path,
        hyper_values,
        labels,
        title,
    )
    return hyper_path


def save_public_corner_plot(output_dir, samples):
    ar_mean = np.asarray(samples["ar_mean"], dtype=float)
    ar_layers = ar_layer_count(samples)
    acm_step_samples, acm_step_labels = acm_step_corner_values(samples)
    ar_median = np.median(ar_mean, axis=0)
    optical_relative = np.divide(
        ar_mean,
        ar_median[None, :],
        out=np.ones_like(ar_mean),
        where=ar_median[None, :] != 0.0,
    )
    values = np.column_stack(
        (
            samples["initial_thickness_nm"],
            samples["mean_rate_nm_day"],
            acm_step_samples,
            optical_relative,
        )
    )
    save_corner_plot(
        output_dir / "numpyro_public_corner.png",
        values,
        (
            "d0 (nm)",
            "mean rate (nm/day)",
        ) + acm_step_labels + ar_parameter_labels(ar_layers, relative=True),
        f"{sample_scope(samples)} posterior; AR parameters normalized to posterior median",
    )


def acm_step_corner_values(samples):
    """Return ACM step percentages and labels for scalar or per-LED archives."""
    log_offsets = np.asarray(samples["acm_step_log_offset"], dtype=float)
    if log_offsets.ndim == 1:
        return 100.0 * np.expm1(log_offsets), ("ACM step (%)",)
    if log_offsets.ndim == 2 and log_offsets.shape[1] == 6:
        return 100.0 * np.expm1(log_offsets), tuple(
            f"ACM LED {led} step (%)" for led in range(1, 7)
        )
    raise ValueError("Archived ACM step posterior must be scalar or six-element")


def npz_scalar(samples, key, default=None):
    if key not in samples:
        return default
    value = np.asarray(samples[key])
    if value.size != 1:
        raise ValueError(f"Archived posterior metadata {key} is not scalar")
    return value.reshape(-1)[0].item()


def save_map_optimization_plot(path, samples):
    """Plot MAP objective progress and the remaining objective gap."""
    if "map_objective_history" not in samples:
        return False
    history = np.asarray(samples["map_objective_history"], dtype=float).reshape(-1)
    if history.size < 2 or not np.all(np.isfinite(history)):
        return False
    iterations = np.arange(history.size)
    final_objective = history[-1]
    objective_floor = max(
        1.0e-12,
        10.0 * np.finfo(float).eps * max(1.0, abs(final_objective)),
    )
    objective_gap = np.maximum(history - final_objective, objective_floor)
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(iterations, history, color="#0072B2", linewidth=1.2)
    axes[0].set_ylabel("Negative log joint")
    axes[0].set_title("MAP objective history")
    axes[0].grid(alpha=0.2)
    axes[1].semilogy(iterations, objective_gap, color="#D55E00", linewidth=1.2)
    axes[1].set_xlabel("Accepted optimizer iterations")
    axes[1].set_ylabel("Objective - final objective")
    axes[1].set_title("Objective gap to final iterate")
    axes[1].grid(alpha=0.2)
    status = npz_scalar(samples, "map_message", "unknown")
    gradient = npz_scalar(samples, "map_scaled_projected_gradient_norm", np.nan)
    figure.suptitle(
        f"{status}; scaled projected gradient={gradient:.3g}",
        fontsize=10,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def resolve_archived_path(value):
    if value is None:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    if not path.is_absolute():
        root_path = ROOT / path
        if root_path.exists():
            return root_path
    return path


def save_tile_fit_scatter_plot(path, data, prediction_quantiles):
    observed = data["sums"] / data["counts"]
    median_prediction = prediction_quantiles[1]
    residual = observed - median_prediction
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10")(data["led"] - 1)
    for led in range(1, 7):
        groups = np.flatnonzero(data["led"] == led)
        axes[0].scatter(
            observed[:, groups].ravel(),
            median_prediction[:, groups].ravel(),
            color=colors[groups[0]],
            alpha=0.18,
            s=7,
            label=f"LED {led}",
        )
    limits = np.asarray([observed.min(), observed.max(), median_prediction.min(), median_prediction.max()])
    axes[0].plot([limits.min(), limits.max()], [limits.min(), limits.max()], "k--", linewidth=1.0)
    axes[0].set_xlabel("Corrected observed tile mean (log change)")
    axes[0].set_ylabel("Modeled tile mean (log change)")
    axes[0].set_title("Observed versus modeled")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].grid(alpha=0.2)

    for led in range(1, 7):
        groups = np.flatnonzero(data["led"] == led)
        axes[1].scatter(
            np.broadcast_to(data["days"][groups], residual[:, groups].shape).ravel(),
            residual[:, groups].ravel(),
            color=colors[groups[0]],
            alpha=0.18,
            s=7,
            label=f"LED {led}",
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Days since first exposure")
    axes[1].set_ylabel("Observed - modeled (log change)")
    axes[1].set_title(f"Residuals; overall RMS={np.sqrt(np.mean(residual**2)):.4g} log")
    axes[1].grid(alpha=0.2)
    figure.suptitle(f"{scope_label(data['sca_names'])} tile-level posterior fit diagnostic")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SINGLE_SCA_OUTPUT)
    parser.add_argument(
        "--tile-statistics",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default=None)
    parser.add_argument(
        "--wavelength-stride",
        type=int,
        default=None,
        help="Sample the QE-supported wavelength grid every N nm; 1 is the production default.",
    )
    parser.add_argument(
        "--wavelength-step-nm",
        type=float,
        default=None,
        help="Use a uniform base grid at this spacing; omit to evaluate exact SED nodes.",
    )
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def _make_plots(args):
    posterior_path = args.output_dir / "posterior_samples.npz"
    tile_posterior_path = args.output_dir / "tile_ar_posterior.csv"
    if not posterior_path.exists() or not tile_posterior_path.exists():
        raise FileNotFoundError(
            f"Expected posterior_samples.npz and tile_ar_posterior.csv in {args.output_dir}"
        )
    samples = np.load(posterior_path)
    archived_sca_names = (
        np.asarray(samples["sca_names"]).astype(str)
        if "sca_names" in samples
        else np.asarray(["WFI01"])
    )
    statistics_path = args.tile_statistics
    if statistics_path is None:
        statistics_path = resolve_archived_path(
            npz_scalar(samples, "statistics_path")
        )
    if statistics_path is None:
        statistics_path = (
            ROOT / "wfi_tile_log_statistics_512"
            if len(archived_sca_names) > 1
            else ROOT / "wfi_tile_log_statistics_512" / "wfi01_tile_log_statistics_512.csv"
        )
    sed_mode = args.sed_mode or npz_scalar(samples, "sed_mode", "core")
    wavelength_stride = args.wavelength_stride
    if wavelength_stride is None:
        wavelength_stride = int(npz_scalar(samples, "wavelength_stride", 1))
    wavelength_step_nm = args.wavelength_step_nm
    if wavelength_step_nm is None:
        archived_step = npz_scalar(samples, "wavelength_step_nm", np.nan)
        wavelength_step_nm = None if np.isnan(archived_step) else float(archived_step)
    archived_min_observation_day = npz_scalar(
        samples, "min_observation_day", np.nan
    )
    min_observation_day = (
        None
        if np.isnan(archived_min_observation_day)
        else float(archived_min_observation_day)
    )
    max_tiles = args.max_tiles
    if max_tiles is None:
        archived_max_tiles = int(npz_scalar(samples, "max_tiles", -1))
        max_tiles = None if archived_max_tiles < 0 else archived_max_tiles
    from fit_wfi_tmm_numpyro import load_tile_statistics

    metadata = load_tile_statistics(
        statistics_path,
        max_groups=np.asarray(samples["ice_thickness"]).shape[-1],
        max_tiles=max_tiles,
        min_observation_day=min_observation_day,
    )
    metadata["qe_columns"] = (
        np.asarray(samples["qe_columns"], dtype=int)
        if "qe_columns" in samples
        else np.arange(len(metadata["sca_names"]), dtype=int)
    )
    qe_plot_path = None
    if bool(npz_scalar(samples, "fit_qe_rbf", 0)):
        qe_plot_path = args.output_dir / "numpyro_qe_modulation.png"
        save_qe_modulation_plot(qe_plot_path, samples, metadata)
    ice_modulation_plot_path = None
    if bool(npz_scalar(samples, "fit_ice_optical_modulation", 0)):
        ice_modulation_plot_path = (
            args.output_dir / "numpyro_ice_optical_modulation.png"
        )
        save_ice_modulation_plot(ice_modulation_plot_path, samples)
    days = metadata["days"]
    repeats = metadata["repeat"]
    rows = load_tile_posterior(tile_posterior_path)
    map_only = bool(npz_scalar(samples, "map_only", 0))
    map_optimization_path = args.output_dir / "numpyro_map_optimization.png"
    has_map_optimization_plot = save_map_optimization_plot(
        map_optimization_path, samples
    )
    trajectory_path = args.output_dir / "numpyro_ice_trajectory.png"
    save_trajectory_plot(trajectory_path, samples, days, args.seed)
    np_trajectory_path = args.output_dir / "numpyro_np_trajectory.png"
    save_np_node_plot(np_trajectory_path, samples, days, repeats)
    tile_map_paths = save_tile_map_plots(args.output_dir, rows)
    tile_rate_map_paths = []
    tile_rate_path = args.output_dir / "tile_rate_posterior.csv"
    if tile_rate_path.exists():
        tile_rate_map_paths = save_tile_rate_map_plots(
            args.output_dir, load_tile_posterior(tile_rate_path)
        )
    hyper_corner_path = None if map_only else save_global_corner_plots(args.output_dir, samples)
    if not map_only:
        save_public_corner_plot(args.output_dir, samples)
    data, prediction_quantiles = load_model_prediction_quantiles(
        args.output_dir,
        statistics_path,
        args.model_path,
        sed_mode,
        wavelength_stride,
        wavelength_step_nm,
        max_tiles,
    )
    acm_step = np.median(
        np.asarray(samples["acm_step_log_offset"], dtype=float), axis=0
    )
    late_acm_step = (
        float(np.median(np.asarray(samples["late_acm_step_log_offset"], dtype=float)))
        if bool(npz_scalar(samples, "fit_late_acm_step", 0))
        else 0.0
    )
    additional_acm_step = (
        float(
            np.median(
                np.asarray(samples["additional_acm_step_log_offset"], dtype=float)
            )
        )
        if bool(npz_scalar(samples, "fit_additional_acm_step", 0))
        else 0.0
    )
    fit_path = args.output_dir / "numpyro_tile_fit.png"
    time_fit_path = args.output_dir / "numpyro_tile_data_model_vs_time.png"
    scatter_path = args.output_dir / "numpyro_tile_fit_scatter.png"
    save_tile_fit_plot(fit_path, data, prediction_quantiles)
    save_tile_data_model_time_plot(
        time_fit_path,
        data,
        prediction_quantiles,
        acm_step=acm_step,
        late_acm_step=late_acm_step,
        additional_acm_step=additional_acm_step,
    )
    save_tile_fit_scatter_plot(scatter_path, data, prediction_quantiles)
    print(f"Wrote {trajectory_path}")
    if has_map_optimization_plot:
        print(f"Wrote {map_optimization_path}")
    if qe_plot_path is not None:
        print(f"Wrote {qe_plot_path}")
    if ice_modulation_plot_path is not None:
        print(f"Wrote {ice_modulation_plot_path}")
    if "np_node_multipliers" in samples:
        print(f"Wrote {np_trajectory_path}")
    for path in tile_map_paths:
        print(f"Wrote {path}")
    for path in tile_rate_map_paths:
        print(f"Wrote {path}")
    if not map_only:
        print(f"Wrote {args.output_dir / 'numpyro_global_corner.png'}")
    if hyper_corner_path is not None:
        print(f"Wrote {hyper_corner_path}")
    if not map_only:
        print(f"Wrote {args.output_dir / 'numpyro_public_corner.png'}")
    print(f"Wrote {fit_path}")
    print(f"Wrote {time_fit_path}")
    print(f"Wrote {scatter_path}")


def make_plots(
    output_dir,
    tile_statistics=None,
    model_path=None,
    sed_mode=None,
    wavelength_stride=None,
    wavelength_step_nm=None,
    max_tiles=None,
    seed=20260922,
):
    """Generate the canonical production plots for an archived fit."""
    return _make_plots(
        argparse.Namespace(
            output_dir=Path(output_dir),
            tile_statistics=(
                None if tile_statistics is None else Path(tile_statistics)
            ),
            model_path=(
                DEFAULT_MODEL_PATH if model_path is None else Path(model_path)
            ),
            sed_mode=sed_mode,
            wavelength_stride=wavelength_stride,
            wavelength_step_nm=wavelength_step_nm,
            max_tiles=max_tiles,
            seed=seed,
        )
    )


def main():
    _make_plots(parse_args())


if __name__ == "__main__":
    main()
