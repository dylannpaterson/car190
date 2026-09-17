"""Plot a NumPyro TMM tile posterior for one or all WFI SCAs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


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
    rates = np.asarray(samples["mean_rate_nm_day"], dtype=float)
    if thickness.ndim != 2 or thickness.shape[1] != len(days):
        raise ValueError("Ice-thickness samples do not match the tile-statistics groups")
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    sorted_thickness = thickness[:, order]
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
    axes[0].set_ylabel("NP ice-thickness multiplier")
    axes[0].set_title("Repeat-node modulation")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    if "np_node_thickness" in samples:
        thickness = np.asarray(samples["np_node_thickness"], dtype=float)
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
    figure.suptitle(f"{sample_scope(samples)} non-parametric ice-thickness modulation")
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
    metrics = (
        ("AR thickness scale", "ar_thickness"),
        ("AR refractive index", "ar_n"),
        ("AR absorption index", "ar_k"),
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

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
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


def load_model_predictions(
    output_dir,
    tile_statistics,
    model_path,
    sed_mode,
    wavelength_stride,
    max_tiles=None,
):
    from fit_wfi_tmm_numpyro import (
        build_prediction_function,
        load_response_context,
        load_tile_statistics,
        require_numpyro,
    )

    jax, jnp, _, _, _, _, _ = require_numpyro()
    jax.config.update("jax_enable_x64", True)
    data = load_tile_statistics(tile_statistics, max_tiles=max_tiles)
    context = load_response_context(model_path, sed_mode, wavelength_stride)
    predictor = build_prediction_function(data, context)
    with np.load(output_dir / "posterior_samples.npz") as samples:
        ice_thickness = np.asarray(samples["ice_thickness"], dtype=float)
        ar_parameters = np.asarray(samples["ar_parameters"], dtype=float)
        acm_step = np.asarray(samples["acm_step_log_offset"], dtype=float)
    if ice_thickness.shape[1] != data["counts"].shape[1]:
        raise ValueError("Posterior groups do not match the tile statistics")
    if ar_parameters.ndim == 4:
        ar_parameters = ar_parameters.reshape(
            ar_parameters.shape[0], -1, ar_parameters.shape[-1]
        )
    if ar_parameters.ndim != 3 or ar_parameters.shape[1] != data["counts"].shape[0]:
        raise ValueError("Posterior tiles do not match the tile statistics")

    batched_predictor = jax.jit(jax.vmap(predictor, in_axes=(0, 0, 0)))
    prediction_chunks = []
    for start in range(0, len(ice_thickness), 16):
        stop = min(start + 16, len(ice_thickness))
        prediction_chunks.append(
            np.asarray(
                batched_predictor(
                    jnp.asarray(ice_thickness[start:stop]),
                    jnp.asarray(ar_parameters[start:stop]),
                    jnp.asarray(acm_step[start:stop]),
                )
            )
        )
    return data, np.concatenate(prediction_chunks, axis=0)


def save_tile_fit_plot(path, data, predictions):
    observed = data["sums"] / data["counts"]
    days = data["days"]
    leds = data["led"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True, constrained_layout=True)
    for led, axis in enumerate(axes.flat, start=1):
        groups = np.flatnonzero(leds == led)
        order = groups[np.argsort(days[groups], kind="stable")]
        x = days[order]
        observed_led = observed[:, order]
        predicted_led = predictions[:, :, order]
        predicted_quantiles = np.percentile(
            predicted_led.reshape(-1, len(order)), [16, 50, 84], axis=0
        )
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
            100.0 * np.expm1(predicted_quantiles[0]),
            100.0 * np.expm1(predicted_quantiles[2]),
            color="#56B4E9",
            alpha=0.35,
            label="Model 16th-84th percentile" if led == 1 else None,
        )
        axis.plot(
            x,
            100.0 * np.expm1(predicted_quantiles[1]),
            color="#0072B2",
            linewidth=2.0,
            label="Model median" if led == 1 else None,
        )
        residual = observed_led - np.percentile(predicted_led, 50, axis=0)
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


def save_tile_data_model_time_plot(path, data, predictions):
    observed = data["sums"] / data["counts"]
    median_prediction = np.percentile(predictions, 50, axis=0)
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
    values = values[np.all(np.isfinite(values), axis=1)]
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
    global_values = np.column_stack(
        (
            samples["initial_thickness_nm"],
            samples["mean_rate_nm_day"],
            100.0 * np.expm1(samples["acm_step_log_offset"]),
            ar_mean,
        )
    )
    save_corner_plot(
        output_dir / "numpyro_global_corner.png",
        global_values,
        (
            "d0 (nm)",
            "mean rate (nm/day)",
            "ACM step (%)",
            "AR scale mean",
            "AR n mean",
            "AR k mean",
        ),
        f"{sample_scope(samples)} global posterior, marginalized over tile parameters",
    )

    if "ar_sigma_raw" in samples:
        hyper_values = np.column_stack(
            (ar_mean, np.asarray(samples["ar_sigma_raw"], dtype=float))
        )
        labels = (
            "AR scale mean",
            "AR n mean",
            "AR k mean",
            "tile sigma scale",
            "tile sigma n",
            "tile sigma k",
        )
        title = f"{sample_scope(samples)} AR population hyperparameters"
    else:
        hyper_values = np.column_stack(
            (
                ar_mean,
                np.asarray(samples["ar_sca_sigma"], dtype=float),
                np.asarray(samples["ar_tile_sigma"], dtype=float),
            )
        )
        labels = (
            "AR scale mean",
            "AR n mean",
            "AR k mean",
            "SCA sigma scale",
            "SCA sigma n",
            "SCA sigma k",
            "tile sigma scale",
            "tile sigma n",
            "tile sigma k",
        )
        title = "All-SCA AR hierarchy hyperparameters"
    save_corner_plot(
        output_dir / "numpyro_ar_hyper_corner.png",
        hyper_values,
        labels,
        title,
    )


def save_public_corner_plot(output_dir, samples):
    ar_mean = np.asarray(samples["ar_mean"], dtype=float)
    ar_median = np.median(ar_mean, axis=0)
    optical_relative = ar_mean / ar_median[None, :]
    values = np.column_stack(
        (
            samples["initial_thickness_nm"],
            samples["mean_rate_nm_day"],
            100.0 * np.expm1(samples["acm_step_log_offset"]),
            optical_relative,
        )
    )
    save_corner_plot(
        output_dir / "numpyro_public_corner.png",
        values,
        (
            "d0 (nm)",
            "mean rate (nm/day)",
            "ACM step (%)",
            "AR thickness / median",
            "AR n / median",
            "AR k / median",
        ),
        f"{sample_scope(samples)} posterior; AR parameters normalized to posterior median",
    )


def save_tile_fit_scatter_plot(path, data, predictions):
    observed = data["sums"] / data["counts"]
    median_prediction = np.percentile(predictions, 50, axis=0)
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
        default=ROOT / "wfi01_tile_log_statistics_512.csv",
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
    parser.add_argument("--sed-mode", choices=("signed", "clipped", "core"), default="core")
    parser.add_argument("--wavelength-stride", type=int, default=20)
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
    statistics_path = args.tile_statistics
    if (
        statistics_path == ROOT / "wfi01_tile_log_statistics_512.csv"
        and "sca_names" in samples
        and len(samples["sca_names"]) > 1
    ):
        statistics_path = ROOT / "wfi_tile_log_statistics_512"
    days, _, repeats = load_group_metadata(statistics_path)
    rows = load_tile_posterior(tile_posterior_path)
    trajectory_path = args.output_dir / "numpyro_ice_trajectory.png"
    save_trajectory_plot(trajectory_path, samples, days, args.seed)
    np_trajectory_path = args.output_dir / "numpyro_np_trajectory.png"
    save_np_node_plot(np_trajectory_path, samples, days, repeats)
    tile_map_paths = save_tile_map_plots(args.output_dir, rows)
    save_global_corner_plots(args.output_dir, samples)
    save_public_corner_plot(args.output_dir, samples)
    data, predictions = load_model_predictions(
        args.output_dir,
        statistics_path,
        args.model_path,
        args.sed_mode,
        args.wavelength_stride,
        args.max_tiles,
    )
    fit_path = args.output_dir / "numpyro_tile_fit.png"
    time_fit_path = args.output_dir / "numpyro_tile_data_model_vs_time.png"
    scatter_path = args.output_dir / "numpyro_tile_fit_scatter.png"
    save_tile_fit_plot(fit_path, data, predictions)
    save_tile_data_model_time_plot(time_fit_path, data, predictions)
    save_tile_fit_scatter_plot(scatter_path, data, predictions)
    print(f"Wrote {trajectory_path}")
    if "np_node_multipliers" in samples:
        print(f"Wrote {np_trajectory_path}")
    for path in tile_map_paths:
        print(f"Wrote {path}")
    print(f"Wrote {args.output_dir / 'numpyro_global_corner.png'}")
    print(f"Wrote {args.output_dir / 'numpyro_ar_hyper_corner.png'}")
    print(f"Wrote {args.output_dir / 'numpyro_public_corner.png'}")
    print(f"Wrote {fit_path}")
    print(f"Wrote {time_fit_path}")
    print(f"Wrote {scatter_path}")


if __name__ == "__main__":
    main()
