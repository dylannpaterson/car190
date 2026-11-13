"""Compare one WFI tile fit with a rate trajectory borrowed from another SCA."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import fit_wfi_tmm_production as production
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    detector_output_root,
    positive_integer,
    tile_directory_entries,
    tile_statistics_path,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET_DETECTOR = "WFI01"
DEFAULT_SOURCE_DETECTOR = "WFI10"
DEFAULT_TILE_ID = 0


def _scalar(samples, name):
    return float(np.asarray(samples[name], dtype=float).reshape(-1)[0])


def _fit_value(samples, name):
    return np.asarray(samples[name], dtype=float).reshape(-1)


def load_source_median_node_rates(source_output_dir, expected_node_count):
    entries = tile_directory_entries(Path(source_output_dir) / "fits")
    complete_tile_ids(
        (tile_id for tile_id, _path in entries),
        label=f"source tile IDs in {source_output_dir}",
    )
    tile_rates = []
    for _tile_id, fit_dir in entries:
        posterior_path = fit_dir / "posterior_samples.npz"
        if not posterior_path.is_file():
            raise FileNotFoundError(f"Missing source posterior: {posterior_path}")
        with np.load(posterior_path, allow_pickle=False) as samples:
            days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            rates = np.asarray(
                samples["node_instantaneous_rate_nm_day"], dtype=float
            ).reshape(-1)
            if rates.size != expected_node_count:
                raise ValueError(
                    f"Unexpected source rate shape in {posterior_path}: "
                    f"{rates.size} versus {expected_node_count}"
                )
            tile_rates.append(rates)

    source_rates = np.median(np.asarray(tile_rates, dtype=float), axis=0)
    if source_rates.size != expected_node_count:
        raise ValueError(
            "Source and target fits have different trajectory-node counts: "
            f"{source_rates.size} versus {expected_node_count}"
        )
    return source_rates


def load_tile_model_inputs(target_fit_dir, target_statistics_path):
    with np.load(target_fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        archive = {name: np.asarray(samples[name]) for name in samples.files}

    data = production.load_tile_statistics(target_statistics_path)
    context = production.load_response_context(data["sca_names"])
    data["qe_columns"] = context["qe_columns"]
    data["response_wavelengths_nm"] = context["wavelengths"]
    data["response_wavelength_count"] = int(context["wavelengths"].size)
    return archive, data, context


def evaluate_rate_trajectory(archive, data, context, log_rate_nodes):
    jax, jnp, _, _ = production.require_numpyro()
    predictor = production.build_prediction_function(data, context)
    ar_raw = jnp.asarray(archive["ar_tile_raw"].reshape(1, -1))
    ar_parameters = production._physical_ar_parameters(ar_raw, jax, jnp)
    row_ar = ar_parameters.reshape((1, 3 * production.AR_LAYERS))
    hgcdte = jnp.asarray(
        [
            _scalar(archive, "absolute_hgcdte_n_log_scale"),
            _scalar(archive, "absolute_hgcdte_k_log_scale"),
        ]
    ).reshape((1, 2))
    ice, rates, _, _ = production._trajectory_state(
        data,
        jnp.asarray(log_rate_nodes).reshape(1, -1),
        jnp.asarray(archive["initial_thickness_log_nm"]),
        jnp.asarray(archive["reset_initial_thickness_log_nm"]),
        jax,
        jnp,
    )
    prediction = predictor(
        ice,
        row_ar,
        hgcdte,
        _scalar(archive, "incidence_angle_deg"),
    )
    return (
        np.asarray(prediction, dtype=float).reshape(-1),
        np.asarray(ice, dtype=float).reshape(-1),
        np.asarray(rates, dtype=float).reshape(-1),
    )


def observed_log_changes(data, archive):
    observed = np.asarray(data["sums"], dtype=float) / np.asarray(
        data["counts"], dtype=float
    )
    observed += _fit_value(archive, "post_reset_tile_log_offset").reshape(-1, 1) * (
        np.asarray(data["days"], dtype=float)[None, :]
        >= float(data["reset_day"])
    )
    return observed.reshape(-1)


def write_comparison_csv(path, data, observed, fitted, counterfactual):
    fields = (
        "group",
        "days",
        "led",
        "repeat",
        "observed_percent",
        "wfi01_model_percent",
        "wfi10_rate_model_percent",
        "wfi01_residual_percent",
        "wfi10_rate_residual_percent",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, group in enumerate(data["groups"]):
            observed_percent = 100.0 * np.expm1(observed[index])
            fitted_percent = 100.0 * np.expm1(fitted[index])
            counterfactual_percent = 100.0 * np.expm1(counterfactual[index])
            writer.writerow(
                {
                    "group": int(group),
                    "days": float(data["days"][index]),
                    "led": int(data["led"][index]),
                    "repeat": int(data["repeat"][index]),
                    "observed_percent": observed_percent,
                    "wfi01_model_percent": fitted_percent,
                    "wfi10_rate_model_percent": counterfactual_percent,
                    "wfi01_residual_percent": 100.0 * np.expm1(observed[index] - fitted[index]),
                    "wfi10_rate_residual_percent": 100.0 * np.expm1(
                        observed[index] - counterfactual[index]
                    ),
                }
            )


def plot_comparison(
    path,
    data,
    observed,
    fitted,
    counterfactual,
    target_detector,
    source_detector,
    tile_id,
    target_mean_rate,
    source_mean_rate,
):
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    observed_percent = 100.0 * np.expm1(observed)
    fitted_percent = 100.0 * np.expm1(fitted)
    counterfactual_percent = 100.0 * np.expm1(counterfactual)
    fitted_residual = 100.0 * np.expm1(observed - fitted)
    counterfactual_residual = 100.0 * np.expm1(observed - counterfactual)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.get_cmap("tab10", 6)
    for led in range(1, 7):
        mask = leds == led
        color = colors(led - 1)
        axes[0].plot(
            days[mask], observed_percent[mask], "o", color=color, ms=3.0, alpha=0.8
        )
        axes[0].plot(days[mask], fitted_percent[mask], color=color, linewidth=1.2)
        axes[0].plot(
            days[mask],
            counterfactual_percent[mask],
            color=color,
            linestyle="--",
            linewidth=1.2,
        )
        axes[1].plot(days[mask], fitted_residual[mask], color=color, linewidth=1.0)
        axes[1].plot(
            days[mask],
            counterfactual_residual[mask],
            color=color,
            linestyle="--",
            linewidth=1.0,
        )

    for axis in axes:
        axis.axvline(float(data["reset_day"]), color="0.35", linestyle=":", linewidth=1.0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Log-change equivalent (%)")
    axes[0].set_title("Observed WFI01 data and fixed-parameter model predictions")
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_ylabel("Observed - model (%)")
    axes[1].set_xlabel("Days since first WFI01 exposure")
    axes[1].set_title("Residuals: solid = fitted WFI01 rate; dashed = WFI10 median rate")

    style_handles = [
        Line2D([], [], marker="o", linestyle="", color="black", label="Observed WFI01"),
        Line2D([], [], color="black", label="WFI01 fitted model"),
        Line2D([], [], color="black", linestyle="--", label="WFI10 median-rate model"),
    ]
    led_handles = [
        Line2D([], [], color=colors(led - 1), label=f"LED {led}")
        for led in range(1, 7)
    ]
    figure.legend(
        handles=style_handles + led_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        f"{target_detector} tile {tile_id:03d}: WFI10 median-rate counterfactual\n"
        f"WFI01 fitted mean rate {target_mean_rate:.4f} nm/day; "
        f"WFI10 detector-median mean rate {source_mean_rate:.4f} nm/day",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.91))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-id", type=int, default=DEFAULT_TILE_ID)
    parser.add_argument("--target-detector", default=DEFAULT_TARGET_DETECTOR)
    parser.add_argument("--source-detector", default=DEFAULT_SOURCE_DETECTOR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tile_id < 0:
        raise ValueError("--tile-id must be non-negative")
    target_detector = args.target_detector.upper()
    source_detector = args.source_detector.upper()
    target_output = detector_output_root(args.root, target_detector, args.tile_size)
    source_output = detector_output_root(args.root, source_detector, args.tile_size)
    target_fit_dir = target_output / "fits" / f"tile_{args.tile_id:03d}"
    target_statistics = tile_statistics_path(
        target_output / "tile_statistics",
        target_detector,
        args.tile_size,
        args.tile_id,
    )
    if not target_statistics.is_file():
        raise FileNotFoundError(f"Missing target tile statistics: {target_statistics}")

    archive, data, context = load_tile_model_inputs(target_fit_dir, target_statistics)
    target_days = np.asarray(data["node_days"], dtype=float)
    source_rate_nodes = load_source_median_node_rates(source_output, target_days.size)
    original_log_rates = archive["np_log_rate_nodes"].reshape(-1)
    fitted, _fitted_ice, _fitted_rates = evaluate_rate_trajectory(
        archive, data, context, original_log_rates
    )
    counterfactual, _counterfactual_ice, _counterfactual_rates = evaluate_rate_trajectory(
        archive, data, context, np.log(source_rate_nodes)
    )
    observed = observed_log_changes(data, archive)

    target_rate = np.asarray(archive["tile_mean_rate_nm_day"], dtype=float).reshape(-1)
    target_mean_rate = float(target_rate[0])
    source_csv = source_output / "tile_rate_posterior.csv"
    with source_csv.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    source_mean_rate = float(
        np.median(
            [float(row["mean_rate_nm_day_median"]) for row in source_rows]
        )
    )

    output = args.output or (
        target_fit_dir / f"{target_detector.lower()}_tile_{args.tile_id:03d}"
        f"_{source_detector.lower()}_median_rate_counterfactual.png"
    )
    comparison_csv = output.with_suffix(".csv")
    metadata_path = output.with_suffix(".json")
    write_comparison_csv(comparison_csv, data, observed, fitted, counterfactual)
    metadata = {
        "target_detector": target_detector,
        "source_detector": source_detector,
        "tile_id": args.tile_id,
        "target_statistics": str(target_statistics.resolve()),
        "target_posterior": str((target_fit_dir / "posterior_samples.npz").resolve()),
        "source_rate_definition": (
            "median over source-detector tiles of node_instantaneous_rate_nm_day; "
            "the full node trajectory is substituted"
        ),
        "target_mean_rate_nm_day": target_mean_rate,
        "source_detector_median_mean_rate_nm_day": source_mean_rate,
        "source_tile_count": len(source_rows),
        "reset_day": float(data["reset_day"]),
        "counterfactual_changes": ["node rate trajectory"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    plot_comparison(
        output,
        data,
        observed,
        fitted,
        counterfactual,
        target_detector,
        source_detector,
        args.tile_id,
        target_mean_rate,
        source_mean_rate,
    )
    print(f"Wrote {output}")
    print(f"Wrote {comparison_csv}")
    print(f"Wrote {metadata_path}")
    print(f"Target fitted mean rate: {target_mean_rate:.9g} nm/day")
    print(f"Source detector median mean rate: {source_mean_rate:.9g} nm/day")
    archived_predictions = []
    with (target_fit_dir / "tile_group_predictions.csv").open(newline="") as handle:
        archived_predictions = np.asarray(
            [float(row["model_log_change"]) for row in csv.DictReader(handle)],
            dtype=float,
        )
    print(
        "Maximum original-model reconstruction difference: "
        f"{np.max(np.abs(fitted - archived_predictions)):.9g} log units"
    )


if __name__ == "__main__":
    main()
