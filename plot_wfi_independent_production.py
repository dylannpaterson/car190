"""Generate production-style plots from independent per-tile WFI fits."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_wfi_tmm_numpyro import (
    save_np_node_plot,
    save_tile_data_model_time_plot,
    save_tile_fit_plot,
    save_tile_fit_scatter_plot,
    save_tile_map_plot,
    save_tile_rate_map_plot,
)


ROOT = Path(__file__).resolve().parent
FIT_SUFFIX = "absolute_substrate_power_hgcdte_prior025_independent_tiles_least_squares"
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
TILE_COUNT = 64


def detector_root(detector, root=ROOT):
    return root / f"{detector.lower()}_{FIT_SUFFIX}"


def tile_fit_dirs(output_root):
    paths = sorted((output_root / "fits").glob("tile_*_least_squares"))
    if len(paths) != TILE_COUNT:
        raise ValueError(
            f"Expected {TILE_COUNT} independent tile fits in {output_root / 'fits'}, "
            f"found {len(paths)}"
        )
    expected = [f"tile_{tile_id:03d}_least_squares" for tile_id in range(TILE_COUNT)]
    if [path.name for path in paths] != expected:
        raise ValueError(f"Independent tile fit IDs are incomplete in {output_root / 'fits'}")
    return paths


def read_csv_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_csv_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_detector_data(detector, output_root, fit_dirs):
    ar_rows = []
    prediction_rows = []
    tile_ids = []
    days = None
    leds = None
    repeats = None
    observed = []
    modeled = []
    thickness = []
    rates = []
    node_rates = []
    node_thickness = []
    node_multipliers = []
    map_histories = []
    node_count = None

    for tile_id, fit_dir in enumerate(fit_dirs):
        with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
            archived_tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)
            if archived_tile_ids.size != 1 or archived_tile_ids[0] != tile_id:
                raise ValueError(f"Unexpected tile IDs in {fit_dir}")
            tile_days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            tile_leds = np.asarray(samples["observation_led"], dtype=int).reshape(-1)
            tile_repeats = np.asarray(samples["observation_repeat"], dtype=int).reshape(-1)
            if days is None:
                days, leds, repeats = tile_days, tile_leds, tile_repeats
            elif not (
                np.array_equal(days, tile_days)
                and np.array_equal(leds, tile_leds)
                and np.array_equal(repeats, tile_repeats)
            ):
                raise ValueError(f"Observation metadata differs in {fit_dir}")

            def one_tile_series(name):
                value = np.asarray(samples[name], dtype=float)
                if value.ndim == 3:
                    value = value[0, 0]
                elif value.ndim == 2:
                    value = value[0]
                return value.reshape(-1)

            thickness.append(one_tile_series("tile_ice_thickness"))
            rates.append(one_tile_series("tile_instantaneous_rate_nm_day"))
            current_node_rates = np.asarray(
                samples["node_instantaneous_rate_nm_day"], dtype=float
            ).reshape(-1)
            current_node_thickness = np.asarray(
                samples["np_node_thickness"], dtype=float
            ).reshape(-1)
            if node_count is None:
                node_count = current_node_rates.size
            if (
                current_node_rates.size != node_count
                or current_node_thickness.size != node_count
            ):
                raise ValueError(f"Node count differs in {fit_dir}")
            node_rates.append(current_node_rates)
            node_thickness.append(current_node_thickness)
            node_multipliers.append(
                np.asarray(samples["np_node_multipliers"], dtype=float).reshape(-1)
            )
            map_histories.append(
                np.asarray(samples["map_objective_history"], dtype=float).reshape(-1)
            )

        current_ar_rows = read_csv_rows(fit_dir / "tile_ar_posterior.csv")
        current_prediction_rows = read_csv_rows(fit_dir / "tile_group_predictions.csv")
        if len(current_ar_rows) != 1:
            raise ValueError(f"Expected one AR row in {fit_dir}")
        ar_rows.extend(current_ar_rows)
        prediction_rows.extend(current_prediction_rows)
        if {int(row["tile_id"]) for row in current_prediction_rows} != {tile_id}:
            raise ValueError(f"Prediction tile IDs differ in {fit_dir}")
        current_prediction_rows.sort(key=lambda row: int(row["group"]))
        observed.append(
            [float(row["observed_log_change"]) for row in current_prediction_rows]
        )
        modeled.append(
            [float(row["model_log_change"]) for row in current_prediction_rows]
        )
        tile_ids.append(tile_id)

    write_csv_rows(output_root / "tile_ar_posterior.csv", ar_rows)
    write_csv_rows(output_root / "tile_group_predictions.csv", prediction_rows)
    observed = np.asarray(observed, dtype=float)
    modeled = np.asarray(modeled, dtype=float)
    if observed.shape != modeled.shape or observed.shape[1] != days.size:
        raise ValueError(f"Unexpected prediction array shape in {output_root}")
    if any(len(values) != days.size for values in thickness + rates):
        raise ValueError(f"Unexpected trajectory length in {output_root}")

    return {
        "detector": detector,
        "days": days,
        "leds": leds,
        "repeats": repeats,
        "observed": observed,
        "modeled": modeled,
        "thickness": np.asarray(thickness, dtype=float),
        "rates": np.asarray(rates, dtype=float),
        "node_rates": np.asarray(node_rates, dtype=float),
        "node_thickness": np.asarray(node_thickness, dtype=float),
        "node_multipliers": np.asarray(node_multipliers, dtype=float),
        "map_histories": map_histories,
        "ar_rows": ar_rows,
        "prediction_rows": prediction_rows,
    }


def production_plot_data(data):
    observation_count = data["observed"].shape[0]
    return {
        "sums": data["observed"],
        "counts": np.ones_like(data["observed"]),
        "days": data["days"],
        "led": data["leds"],
        "repeat": data["repeats"],
        "step_design": np.zeros(data["days"].size),
        "late_step_design": np.zeros(data["days"].size),
        "additional_step_design": np.zeros(data["days"].size),
        "sca_names": np.asarray([data["detector"]]),
        "fixed_corrections_applied": True,
        "n_tiles": observation_count,
    }


def save_independent_trajectory_plot(path, data, reset_day):
    days = data["days"]
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = plt.get_cmap("viridis", TILE_COUNT)
    for tile_id, color_index in enumerate(range(TILE_COUNT)):
        color = colors(color_index)
        axes[0].plot(
            sorted_days,
            data["thickness"][tile_id, order],
            color=color,
            alpha=0.65,
            linewidth=0.8,
        )
        axes[1].plot(
            sorted_days,
            data["rates"][tile_id, order],
            color=color,
            alpha=0.65,
            linewidth=0.8,
        )
    axes[0].plot(
        sorted_days,
        np.mean(data["thickness"][:, order], axis=0),
        color="black",
        linewidth=1.8,
        label="Mean across tiles",
    )
    axes[1].plot(
        sorted_days,
        np.mean(data["rates"][:, order], axis=0),
        color="black",
        linewidth=1.8,
        label="Mean across tiles",
    )
    for axis in axes:
        axis.axvline(
            reset_day,
            color="0.35",
            linestyle=":",
            linewidth=0.9,
            alpha=0.8,
            label="Ice reset",
        )
        axis.set_xlabel("Days since first exposure")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Ice thickness (nm)")
    axes[0].set_title("Per-tile ice trajectories (64 tiles)")
    axes[1].set_ylabel("Instantaneous accumulation rate (nm/day)")
    axes[1].set_title("Per-tile instantaneous rates (64 tiles)")
    figure.suptitle(f"{data['detector']} independent per-tile TMM fits")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_independent_map_plot(path, data):
    histories = data["map_histories"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for history in histories:
        iterations = np.arange(history.size)
        axes[0].plot(iterations, history, color="#0072B2", alpha=0.18, linewidth=0.7)
        gap = np.maximum(history - history[-1], 1.0e-12)
        axes[1].semilogy(iterations, gap, color="#D55E00", alpha=0.18, linewidth=0.7)
    axes[0].set_ylabel("Negative log joint")
    axes[0].set_title("Independent tile MAP objective histories")
    axes[1].set_xlabel("Accepted optimizer iterations")
    axes[1].set_ylabel("Objective - tile final objective")
    axes[1].set_title("Objective gap to each tile's final iterate")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"{data['detector']} independent per-tile MAP diagnostics")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_detector_plots(detector, root=ROOT):
    output_root = detector_root(detector, root)
    fit_dirs = tile_fit_dirs(output_root)
    data = load_detector_data(detector, output_root, fit_dirs)
    plot_data = production_plot_data(data)
    prediction_quantiles = np.stack(
        (data["modeled"], data["modeled"], data["modeled"])
    )

    save_independent_trajectory_plot(
        output_root / "numpyro_ice_trajectory.png",
        data,
        reset_day=15.064405023148149,
    )
    node_samples = {
        "sca_names": np.asarray([detector]),
        "fully_np_trajectory": np.asarray(1),
        "node_instantaneous_rate_nm_day": np.mean(
            data["node_rates"], axis=0, keepdims=True
        ),
        "np_node_thickness": np.mean(
            data["node_thickness"], axis=0, keepdims=True
        ),
        "np_node_multipliers": np.mean(
            data["node_multipliers"], axis=0, keepdims=True
        ),
        "trajectory_rate_prior_mean_nm_day": np.asarray(5.0),
    }
    save_np_node_plot(
        output_root / "numpyro_np_trajectory.png",
        node_samples,
        data["days"],
        data["repeats"],
    )
    save_tile_map_plot(output_root / "numpyro_tile_ar_maps.png", data["ar_rows"])
    save_tile_rate_map_plot(
        output_root / f"numpyro_tile_rate_maps_{detector}.png",
        read_csv_rows(output_root / "tile_rate_posterior.csv"),
    )
    save_tile_fit_plot(
        output_root / "numpyro_tile_fit.png", plot_data, prediction_quantiles
    )
    save_tile_data_model_time_plot(
        output_root / "numpyro_tile_data_model_vs_time.png",
        plot_data,
        prediction_quantiles,
    )
    save_tile_fit_scatter_plot(
        output_root / "numpyro_tile_fit_scatter.png",
        plot_data,
        prediction_quantiles,
    )
    save_independent_map_plot(output_root / "numpyro_map_optimization.png", data)
    return output_root


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        "--output-dir",
        dest="root",
        type=Path,
        default=ROOT,
        help="Parent folder containing detector independent-fit outputs.",
    )
    parser.add_argument(
        "--detector",
        action="append",
        help="Generate plots for one detector; omit to generate all WFI01-WFI18.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detectors = [value.upper() for value in args.detector] if args.detector else DETECTORS
    unknown = sorted(set(detectors).difference(DETECTORS))
    if unknown:
        raise ValueError(f"Unknown detectors: {unknown}")
    for detector in detectors:
        print(f"Generating independent production-style plots for {detector}", flush=True)
        output_root = save_detector_plots(detector, args.root)
        print(f"Wrote plots under {output_root}", flush=True)


if __name__ == "__main__":
    main()
