"""Plot a NumPyro TMM tile posterior for one or all WFI SCAs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fit_wfi_tmm_numpyro import (
    ASW_OPTICAL_CONSTANTS_URL,
    QE_RBF_NODE_WAVELENGTHS_NM,
    QE_RBF_WIDTH_NM,
    PREDICTION_ROW_CHUNK_SIZE,
)
from assess_rcs_spectral_identifiability import QE_URL, load_qe


ROOT = Path(__file__).resolve().parent
DEFAULT_SINGLE_SCA_OUTPUT = ROOT / "wfi01_tmm_numpyro_tile_pilot"


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
    if thickness.ndim == 3:
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
    has_rate_trajectory = (
        instantaneous_rates is not None
        and instantaneous_rates.ndim == 2
        and instantaneous_rates.shape == thickness.shape
    )
    quantiles = np.percentile(sorted_thickness, [16, 50, 84], axis=0)
    rng = np.random.default_rng(seed)
    draw_count = min(100, len(sorted_thickness))
    draw_indices = rng.choice(len(sorted_thickness), draw_count, replace=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
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
    axes[0].plot(sorted_days, quantiles[1], color="#0072B2", linewidth=2.0, label="Median")
    axes[0].set_xlabel("Days since first exposure")
    axes[0].set_ylabel("Ice thickness (nm)")
    axes[0].set_title("Shared ice trajectory")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    if has_rate_trajectory:
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


def save_np_node_plot(path, samples, days, repeats):
    if "np_node_multipliers" not in samples:
        return
    repeat_values = np.asarray(sorted(set(repeats.tolist())), dtype=int)
    node_days = np.asarray(
        [np.mean(days[repeats == repeat]) for repeat in repeat_values],
        dtype=float,
    )
    multipliers = np.asarray(samples["np_node_multipliers"], dtype=float)
    multiplier_quantiles = np.percentile(multipliers, [16, 50, 84], axis=0)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].fill_between(
        node_days,
        multiplier_quantiles[0],
        multiplier_quantiles[2],
        color="#56B4E9",
        alpha=0.35,
        label="16th-84th percentile",
    )
    axes[0].plot(
        node_days,
        multiplier_quantiles[1],
        color="#0072B2",
        linewidth=2.0,
        label="Median",
    )
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Days since first exposure")
    axes[0].set_ylabel("NP instantaneous-rate multiplier")
    axes[0].set_title("Repeat-node rate modulation")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    if "np_node_thickness" in samples:
        thickness = np.asarray(samples["np_node_thickness"], dtype=float)
        if thickness.ndim == 3:
            thickness = np.mean(thickness, axis=1)
        thickness_quantiles = np.percentile(thickness, [16, 50, 84], axis=0)
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
        axes[1].set_title("NP-modulated node thickness")
        axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Days since first exposure")
    axes[1].grid(alpha=0.2)
    figure.suptitle(f"{sample_scope(samples)} non-parametric rate modulation")
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
        fit_hgcdte = bool(npz_scalar(samples, "fit_hgcdte", 0))
        hgcdte_parameters = None
        if fit_hgcdte:
            hgcdte_parameters = np.column_stack(
                (
                    np.asarray(samples["hgcdte_n"], dtype=float),
                    np.asarray(samples["hgcdte_k"], dtype=float),
                )
            )
        fit_conductive_layer = bool(
            npz_scalar(samples, "fit_conductive_layer", 0)
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
    data = load_tile_statistics(
        tile_statistics,
        max_tiles=max_tiles,
        max_groups=ice_thickness.shape[-1],
    )
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
        ASW_OPTICAL_CONSTANTS_URL if use_asw_optical_constants else None,
    )
    context["qe_rbf_node_wavelengths_nm"] = qe_rbf_node_wavelengths_nm
    context["qe_rbf_width_nm"] = qe_rbf_width_nm
    context["use_asw_optical_constants"] = use_asw_optical_constants
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
        full_data = load_tile_statistics(tile_statistics)
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
        ar_parameters = ar_parameters.reshape(
            ar_parameters.shape[0], len(data["sca_names"]), posterior_tiles, -1
        )
        ar_parameters = ar_parameters[:, :, selected_indices, :].reshape(
            ar_parameters.shape[0], -1, ar_parameters.shape[-1]
        )
    if ar_parameters.shape[1] != data["counts"].shape[0]:
        raise ValueError("Posterior tiles do not match the tile statistics")

    return (
        data,
        context,
        ice_thickness,
        ar_parameters,
        acm_step,
        hgcdte_parameters,
        fit_hgcdte,
        conductive_parameters,
        fit_conductive_layer,
        qe_rbf_parameters,
        fit_qe_rbf,
        jax,
        jnp,
        build_prediction_function,
    )


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
        fit_hgcdte,
        conductive_parameters,
        fit_conductive_layer,
        qe_rbf_parameters,
        fit_qe_rbf,
        jax,
        jnp,
        build_prediction_function,
    ) = _load_prediction_inputs(
        output_dir,
        tile_statistics,
        model_path,
        sed_mode,
        wavelength_stride,
        wavelength_step_nm,
        max_tiles,
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
                fit_conductive_layer=fit_conductive_layer,
                fit_qe_rbf=fit_qe_rbf,
                qe_rbf_node_wavelengths_nm=context["qe_rbf_node_wavelengths_nm"],
                qe_rbf_width_nm=context["qe_rbf_width_nm"],
                use_asw_optical_constants=context["use_asw_optical_constants"],
            )
            predictor_in_axes = [0, 0, 0]
            if fit_hgcdte:
                predictor_in_axes.append(0)
            elif fit_conductive_layer:
                predictor_in_axes.append(None)
            if fit_conductive_layer:
                predictor_in_axes.append(0)
            if fit_qe_rbf:
                if not fit_hgcdte and not fit_conductive_layer:
                    predictor_in_axes.append(None)
                if not fit_conductive_layer:
                    predictor_in_axes.append(None)
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
            if fit_hgcdte:
                prediction_args += (
                    jnp.asarray(hgcdte_parameters[start:stop]),
                )
            elif fit_conductive_layer:
                prediction_args += (None,)
            if fit_conductive_layer:
                prediction_args += (
                    jnp.asarray(conductive_parameters[start:stop]),
                )
            if fit_qe_rbf:
                if not fit_hgcdte and not fit_conductive_layer:
                    prediction_args += (None,)
                if not fit_conductive_layer:
                    prediction_args += (None,)
                prediction_args += (
                    jnp.asarray(qe_rbf_parameters[start:stop]),
                )
            row_chunks.append(np.asarray(row_batched_predictor(*prediction_args)))
        prediction_chunks.append(np.concatenate(row_chunks, axis=1))
    return data, np.concatenate(prediction_chunks, axis=0)


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
        fit_hgcdte,
        conductive_parameters,
        fit_conductive_layer,
        qe_rbf_parameters,
        fit_qe_rbf,
        jax,
        jnp,
        build_prediction_function,
    ) = _load_prediction_inputs(
        output_dir,
        tile_statistics,
        model_path,
        sed_mode,
        wavelength_stride,
        wavelength_step_nm,
        max_tiles,
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
            fit_conductive_layer=fit_conductive_layer,
            fit_qe_rbf=fit_qe_rbf,
            qe_rbf_node_wavelengths_nm=context["qe_rbf_node_wavelengths_nm"],
            qe_rbf_width_nm=context["qe_rbf_width_nm"],
            use_asw_optical_constants=context["use_asw_optical_constants"],
        )
        predictor_in_axes = [0, 0, 0]
        if fit_hgcdte:
            predictor_in_axes.append(0)
        elif fit_conductive_layer:
            predictor_in_axes.append(None)
        if fit_conductive_layer:
            predictor_in_axes.append(0)
        if fit_qe_rbf:
            if not fit_hgcdte and not fit_conductive_layer:
                predictor_in_axes.append(None)
            if not fit_conductive_layer:
                predictor_in_axes.append(None)
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
            if fit_hgcdte:
                prediction_args += (
                    jnp.asarray(hgcdte_parameters[start:stop]),
                )
            elif fit_conductive_layer:
                prediction_args += (None,)
            if fit_conductive_layer:
                prediction_args += (
                    jnp.asarray(conductive_parameters[start:stop]),
                )
            if fit_qe_rbf:
                if not fit_hgcdte and not fit_conductive_layer:
                    prediction_args += (None,)
                if not fit_conductive_layer:
                    prediction_args += (None,)
                prediction_args += (
                    jnp.asarray(qe_rbf_parameters[start:stop]),
                )
            sample_predictions.append(
                np.asarray(row_batched_predictor(*prediction_args))
            )
        row_predictions = np.concatenate(sample_predictions, axis=0)
        quantile_chunks.append(
            np.percentile(row_predictions, [16.0, 50.0, 84.0], axis=0)
        )
    return data, np.concatenate(quantile_chunks, axis=1)


def save_tile_fit_plot(path, data, prediction_quantiles):
    observed = data["sums"] / data["counts"]
    days = data["days"]
    leds = data["led"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True, constrained_layout=True)
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
        axis.scatter(
            np.broadcast_to(x, observed_led.shape).ravel(),
            (100.0 * np.expm1(observed_led)).ravel(),
            color="#444444",
            alpha=0.18,
            s=7,
            label="Tile observations" if led == 1 else None,
        )
        axis.fill_between(
            x,
            100.0 * np.expm1(predicted_curve_quantiles[0]),
            100.0 * np.expm1(predicted_curve_quantiles[2]),
            color="#56B4E9",
            alpha=0.35,
            label=model_band_label if led == 1 else None,
        )
        axis.plot(
            x,
            100.0 * np.expm1(predicted_curve_quantiles[1]),
            color="#0072B2",
            linewidth=2.0,
            label="Model median" if led == 1 else None,
        )
        residual = observed_led - prediction_quantiles[1][:, order]
        rms = np.sqrt(np.mean(residual**2))
        axis.set_title(f"LED {led}; tile RMS={rms:.4g} log")
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Days since first exposure")
    axes[0, 0].set_ylabel("Relative change (%)")
    axes[1, 0].set_ylabel("Relative change (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{scope_label(data['sca_names'])} tile observations versus posterior TMM model")
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


def save_tile_data_model_time_plot(path, data, prediction_quantiles):
    observed = data["sums"] / data["counts"]
    median_prediction = prediction_quantiles[1]
    days = data["days"]
    leds = data["led"]
    figure, axes = plt.subplots(
        2, 3, figsize=(15, 8), sharex=False, sharey=True, constrained_layout=True
    )
    for led, axis in enumerate(axes.flat, start=1):
        groups = np.flatnonzero(leds == led)
        order = groups[np.argsort(days[groups], kind="stable")]
        x = days[order]
        observed_led = observed[:, order]
        model_led = median_prediction[:, order]
        for tile in range(observed_led.shape[0]):
            axis.scatter(
                x,
                100.0 * np.expm1(observed_led[tile]),
                color="#555555",
                alpha=0.16,
                s=7,
                linewidths=0,
            )
            axis.plot(
                x,
                100.0 * np.expm1(model_led[tile]),
                color="#0072B2",
                alpha=0.22,
                linewidth=0.8,
            )
        residual = observed_led - model_led
        axis.set_title(f"LED {led}; RMS={np.sqrt(np.mean(residual**2)):.4g} log")
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_xlabel("Days since first exposure")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Relative change (%)")
    axes[1, 0].set_ylabel("Relative change (%)")
    axes[0, 0].scatter([], [], color="#555555", s=18, label="Observed tile data")
    axes[0, 0].plot([], [], color="#0072B2", linewidth=2.0, label="Tile posterior model")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{scope_label(data['sca_names'])} tile data with the modeled curves overlaid")
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
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
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
    axes[0].set_xlabel("Observed tile mean (log change)")
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
        default=Path(
            "/Users/paterson.35/Documents/"
            "rgespit-comissioning-ice-simulations/data/ice_optical_constants/"
            "spectral_response_model_SCA1.ecsv"
        ),
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


def main():
    args = parse_args()
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
            else ROOT / "wfi01_tile_log_statistics_512.csv"
        )
    sed_mode = args.sed_mode or npz_scalar(samples, "sed_mode", "core")
    wavelength_stride = args.wavelength_stride
    if wavelength_stride is None:
        wavelength_stride = int(npz_scalar(samples, "wavelength_stride", 1))
    wavelength_step_nm = args.wavelength_step_nm
    if wavelength_step_nm is None:
        archived_step = npz_scalar(samples, "wavelength_step_nm", np.nan)
        wavelength_step_nm = None if np.isnan(archived_step) else float(archived_step)
    max_tiles = args.max_tiles
    if max_tiles is None:
        archived_max_tiles = int(npz_scalar(samples, "max_tiles", -1))
        max_tiles = None if archived_max_tiles < 0 else archived_max_tiles
    from fit_wfi_tmm_numpyro import load_tile_statistics

    metadata = load_tile_statistics(
        statistics_path,
        max_groups=np.asarray(samples["ice_thickness"]).shape[-1],
        max_tiles=max_tiles,
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
    days = metadata["days"]
    repeats = metadata["repeat"]
    rows = load_tile_posterior(tile_posterior_path)
    map_only = bool(npz_scalar(samples, "map_only", 0))
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
    fit_path = args.output_dir / "numpyro_tile_fit.png"
    time_fit_path = args.output_dir / "numpyro_tile_data_model_vs_time.png"
    scatter_path = args.output_dir / "numpyro_tile_fit_scatter.png"
    save_tile_fit_plot(fit_path, data, prediction_quantiles)
    save_tile_data_model_time_plot(time_fit_path, data, prediction_quantiles)
    save_tile_fit_scatter_plot(scatter_path, data, prediction_quantiles)
    print(f"Wrote {trajectory_path}")
    if qe_plot_path is not None:
        print(f"Wrote {qe_plot_path}")
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


if __name__ == "__main__":
    main()
