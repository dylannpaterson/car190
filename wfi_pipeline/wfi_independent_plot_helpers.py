"""Plot helpers for independent per-tile production archives."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from wfi_production_plot_annotations import (
    annotate_time_axis,
    event_legend_handles,
    non_annotation_legend_handles,
    split_time_series_at_reset,
    split_time_series_at_resets,
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


def npz_scalar(samples, key, default=None):
    if key not in samples:
        return default
    value = np.asarray(samples[key])
    if value.size != 1:
        raise ValueError(f"Archived plot metadata {key} is not scalar")
    return value.reshape(-1)[0].item()


def save_np_node_plot(
    path,
    samples,
    days,
    repeats,
    time_annotations=None,
    reset_days=None,
    source_label=None,
    node_title=None,
    thickness_title=None,
    percentile_label="16th-84th percentile",
    rate_y_limit=None,
    date_tick_interval_days=1,
):
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
        default_node_title = "Fully non-parametric observation-epoch-node rate"
    else:
        multipliers = np.asarray(samples["np_node_multipliers"], dtype=float)
        node_value_quantiles = np.percentile(multipliers, [16, 50, 84], axis=0)
        node_ylabel = "NP instantaneous-rate multiplier"
        default_node_title = "Observation-epoch-node rate modulation"

    thickness_quantiles = None
    if "np_node_thickness" in samples:
        thickness = np.asarray(samples["np_node_thickness"], dtype=float)
        if thickness.ndim == 3:
            thickness = np.mean(thickness, axis=1)
        thickness_quantiles = np.percentile(thickness, [16, 50, 84], axis=0)
    if reset_days is None:
        reset_day = (
            None if time_annotations is None else time_annotations.physical_reset_day
        )
        reset_days = None if reset_day is None else (reset_day,)
    if thickness_quantiles is None:
        node_days, node_value_quantiles = split_time_series_at_resets(
            node_days,
            node_value_quantiles,
            reset_days=reset_days,
        )
    else:
        node_days, node_value_quantiles, thickness_quantiles = (
            split_time_series_at_resets(
                node_days,
                node_value_quantiles,
                thickness_quantiles,
                reset_days=reset_days,
            )
        )

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].fill_between(
        node_days,
        node_value_quantiles[0],
        node_value_quantiles[2],
        color="#56B4E9",
        alpha=0.35,
        label=percentile_label,
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
    axes[0].set_title(
        node_title or default_node_title
    )
    axes[0].grid(alpha=0.2)
    if rate_y_limit is not None:
        axes[0].set_ylim(top=float(rate_y_limit))

    if thickness_quantiles is not None:
        axes[1].fill_between(
            node_days,
            thickness_quantiles[0],
            thickness_quantiles[2],
            color="#E69F00",
            alpha=0.35,
            label=percentile_label,
        )
        axes[1].plot(
            node_days,
            thickness_quantiles[1],
            color="#D55E00",
            linewidth=2.0,
            label="Median",
        )
        axes[1].set_ylabel("Ice thickness (nm)")
        axes[1].set_title(
            thickness_title or "Ice thickness at observation-epoch nodes"
        )
    axes[1].set_xlabel("Days since first exposure")
    axes[1].grid(alpha=0.2)
    if time_annotations is not None:
        annotate_time_axis(
            axes[0],
            time_annotations,
            date_tick_interval_days=date_tick_interval_days,
        )
        annotate_time_axis(
            axes[1],
            time_annotations,
            date_tick_interval_days=date_tick_interval_days,
        )
    for axis in axes:
        handles, labels = non_annotation_legend_handles(axis, time_annotations)
        if handles:
            axis.legend(handles, labels, frameon=False, fontsize=8)
    figure.suptitle(
        (
            f"{scope_label(samples['sca_names'])} {source_label}"
            if source_label
            else f"{scope_label(samples['sca_names'])} "
            + (
                "fully non-parametric ice trajectory"
                if fully_np_trajectory
                else "non-parametric rate modulation"
            )
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
        None if time_annotations is None else time_annotations.physical_reset_day
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
        axis.set_title(f"LED {led}; tile RMS={np.sqrt(np.mean(residual**2)):.4g} log")
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Days since first exposure")
    if time_annotations is not None:
        for axis in axes.flat:
            annotate_time_axis(axis, time_annotations)
    axes[0, 0].set_ylabel("Relative change (%)")
    axes[1, 0].set_ylabel("Relative change (%)")
    handles, labels = non_annotation_legend_handles(axes[0, 0], time_annotations)
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


def save_tile_data_model_time_plot(
    path,
    data,
    prediction_quantiles,
    acm_step=0.0,
    late_acm_step=0.0,
    additional_acm_step=0.0,
    time_annotations=None,
):
    """Plot every LED after ACM handling; fixed corrections are already in data."""
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
    additional_acm_step = float(np.asarray(additional_acm_step).reshape(-1)[0])
    acm_correction += additional_acm_step * additional_step_design
    observed -= acm_correction[None, :]
    prediction_quantiles -= acm_correction[None, None, :]

    figure, axis = plt.subplots(figsize=(13, 7))
    reset_day = (
        None if time_annotations is None else time_annotations.physical_reset_day
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
        Line2D([], [], color=color_by_led[led], linewidth=2.0, label=f"LED {led}")
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
    axis.set_ylabel("Relative change after fixed gain/ACM correction (%)")
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
        "fixed gain/ACM corrections applied to data",
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
    limits = np.asarray(
        [observed.min(), observed.max(), median_prediction.min(), median_prediction.max()]
    )
    axes[0].plot(
        [limits.min(), limits.max()],
        [limits.min(), limits.max()],
        "k--",
        linewidth=1.0,
    )
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
