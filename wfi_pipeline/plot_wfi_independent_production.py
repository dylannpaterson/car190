"""Generate production-style plots from independent per-tile WFI fits."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wfi_independent_plot_helpers import (
    save_np_node_plot,
    save_tile_data_model_time_plot,
    save_tile_fit_plot,
    save_tile_fit_scatter_plot,
    save_tile_map_plot,
    save_tile_rate_map_plot,
)
from wfi_production_plot_annotations import (
    annotate_time_axis,
    load_time_annotations,
    event_legend_handles,
    non_annotation_legend_handles,
    split_time_series_at_resets,
)
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    complete_tile_ids,
    corrected_statistics_path,
    detector_output_root,
    positive_integer,
    tile_directory_entries,
)


ROOT = Path(__file__).resolve().parents[1]
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_RESET_DAY = 12.907397928240743


def detector_root(detector, root=ROOT, tile_size=DEFAULT_TILE_SIZE):
    return detector_output_root(root, detector, tile_size)


def tile_fit_dirs(output_root):
    entries = [
        (tile_id, path)
        for tile_id, path in tile_directory_entries(output_root / "fits")
        if (path / "posterior_samples.npz").is_file()
    ]
    complete_tile_ids(
        (tile_id for tile_id, _path in entries),
        label=f"independent tile fit IDs in {output_root / 'fits'}",
    )
    return [path for _tile_id, path in entries]


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
    reset_days = None
    reset_event_names = None

    for tile_id, fit_dir in enumerate(fit_dirs):
        with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
            archived_tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)
            if archived_tile_ids.size != 1 or archived_tile_ids[0] != tile_id:
                raise ValueError(f"Unexpected tile IDs in {fit_dir}")
            tile_days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            tile_leds = np.asarray(samples["observation_led"], dtype=int).reshape(-1)
            tile_repeats = np.asarray(samples["observation_repeat"], dtype=int).reshape(-1)
            if "reset_event_days" in samples:
                tile_reset_days = np.asarray(samples["reset_event_days"], dtype=float).reshape(-1)
                tile_reset_names = np.asarray(samples["reset_event_names"]).astype(str).reshape(-1)
            else:
                tile_reset_days = np.asarray(
                    samples.get("ice_reset_days", [samples.get("ice_reset_day", DEFAULT_RESET_DAY)]),
                    dtype=float,
                ).reshape(-1)
                tile_reset_names = np.asarray(
                    ["second_decon"] + ["third_decon"] * max(0, tile_reset_days.size - 1)
                ).astype(str)
            if days is None:
                days, leds, repeats = tile_days, tile_leds, tile_repeats
                reset_days, reset_event_names = tile_reset_days, tile_reset_names
            elif not (
                np.array_equal(days, tile_days)
                and np.array_equal(leds, tile_leds)
                and np.array_equal(repeats, tile_repeats)
            ):
                raise ValueError(f"Observation metadata differs in {fit_dir}")
            elif not (
                np.array_equal(reset_days, tile_reset_days)
                and np.array_equal(reset_event_names, tile_reset_names)
            ):
                raise ValueError(f"Reset-event metadata differs in {fit_dir}")

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
        "reset_days": np.asarray(reset_days, dtype=float),
        "reset_event_names": np.asarray(reset_event_names).astype(str),
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


def save_independent_trajectory_plot(
    path, data, reset_day=None, time_annotations=None, reset_days=None
):
    days = data["days"]
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    sorted_thickness = data["thickness"][:, order]
    sorted_rates = data["rates"][:, order]
    if reset_days is None:
        reset_days = (reset_day,) if reset_day is not None else ()
    plot_days, sorted_thickness, sorted_rates = split_time_series_at_resets(
        sorted_days,
        sorted_thickness,
        sorted_rates,
        reset_days=reset_days,
    )
    mean_thickness = np.mean(data["thickness"][:, order], axis=0)
    mean_rates = np.mean(data["rates"][:, order], axis=0)
    _, mean_thickness, mean_rates = split_time_series_at_resets(
        sorted_days,
        mean_thickness,
        mean_rates,
        reset_days=reset_days,
    )
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    tile_count = data["thickness"].shape[0]
    colors = plt.get_cmap("viridis", tile_count)
    for tile_id, color_index in enumerate(range(tile_count)):
        color = colors(color_index)
        axes[0].plot(
            plot_days,
            sorted_thickness[tile_id],
            color=color,
            alpha=0.65,
            linewidth=0.8,
        )
        axes[1].plot(
            plot_days,
            sorted_rates[tile_id],
            color=color,
            alpha=0.65,
            linewidth=0.8,
        )
    axes[0].plot(
        plot_days,
        mean_thickness,
        color="black",
        linewidth=1.8,
        label="Mean across tiles",
    )
    axes[1].plot(
        plot_days,
        mean_rates,
        color="black",
        linewidth=1.8,
        label="Mean across tiles",
    )
    for index, axis in enumerate(axes):
        if time_annotations is not None:
            annotate_time_axis(axis, time_annotations)
        for reset_index, reset_boundary in enumerate(reset_days):
            axis.axvline(
                reset_boundary,
                color="0.35",
                linestyle=":",
                linewidth=0.9,
                alpha=0.8,
                label=(
                    "Decon2"
                    if reset_index == 0
                    else "Third decon"
                    if reset_index == 1
                    else f"Reset {reset_index + 1}"
                ),
            )
        axis.set_xlabel("Days since first exposure" if index == len(axes) - 1 else "")
        axis.grid(alpha=0.2)
        handles, labels = non_annotation_legend_handles(axis, time_annotations)
        if handles:
            axis.legend(handles, labels, frameon=False, fontsize=8)
    axes[0].set_ylabel("Ice thickness (nm)")
    axes[0].set_title(f"Per-tile ice trajectories ({tile_count} tiles)")
    axes[1].set_ylabel("Instantaneous accumulation rate (nm/day)")
    axes[1].set_title(f"Per-tile instantaneous rates ({tile_count} tiles)")
    figure.suptitle(f"{data['detector']} independent per-tile TMM fits", y=0.98)
    if time_annotations is not None:
        figure.legend(
            handles=event_legend_handles(time_annotations),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=8,
            frameon=False,
            fontsize=7,
        )
    figure.subplots_adjust(top=0.88, bottom=0.15, hspace=0.24)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_rate_histogram(path, data, reset_day):
    days = np.asarray(data["days"], dtype=float)
    rates = np.asarray(data["rates"], dtype=float)
    pre_mask = days < reset_day
    post_mask = days >= reset_day
    if not np.any(pre_mask) or not np.any(post_mask):
        raise ValueError(f"Reset day does not split observations for {data['detector']}")

    pre_rates = rates[:, pre_mask].reshape(-1)
    post_rates = rates[:, post_mask].reshape(-1)
    pre_rates = pre_rates[np.isfinite(pre_rates)]
    post_rates = post_rates[np.isfinite(post_rates)]
    latest_index = int(np.argmax(days))
    current_rates = rates[:, latest_index]
    current_rates = current_rates[np.isfinite(current_rates)]
    if pre_rates.size == 0 or post_rates.size == 0 or current_rates.size == 0:
        raise ValueError(f"No finite instantaneous rates for {data['detector']}")

    all_rates = np.concatenate((pre_rates, post_rates))
    rate_min = float(np.min(all_rates))
    rate_max = float(np.max(all_rates))
    if np.isclose(rate_min, rate_max):
        half_width = max(abs(rate_min) * 0.05, 0.1)
        bins = np.linspace(rate_min - half_width, rate_max + half_width, 21)
    else:
        bins = np.histogram_bin_edges(all_rates, bins=32)
    current_rate = float(np.median(current_rates))

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.hist(
        pre_rates,
        bins=bins,
        density=True,
        color="#0072B2",
        alpha=0.60,
        label=f"Pre-Decon2 (N={pre_rates.size:,})",
    )
    axis.hist(
        post_rates,
        bins=bins,
        density=True,
        color="#D55E00",
        alpha=0.55,
        label=f"Post-Decon2 (N={post_rates.size:,})",
    )
    axis.axvline(
        current_rate,
        color="black",
        linewidth=2.0,
        label=f"Current fitted median: {current_rate:.3f} nm/day",
    )
    axis.set_xlabel("Instantaneous ice accumulation rate (nm/day)")
    axis.set_ylabel("Probability density")
    axis.set_title(
        f"{data['detector']} instantaneous ice accumulation rates by Decon2 epoch\n"
        f"Decon2 day {reset_day:.3f}; current epoch day {days[latest_index]:.3f}"
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
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


def save_detector_plots(
    detector,
    root=ROOT,
    reset_day=DEFAULT_RESET_DAY,
    output_root=None,
    statistics_path=None,
    tile_size=DEFAULT_TILE_SIZE,
    reset_days=None,
):
    output_root = (
        detector_root(detector, root, tile_size)
        if output_root is None
        else Path(output_root)
    )
    fit_dirs = tile_fit_dirs(output_root)
    if statistics_path is None:
        statistics_path = corrected_statistics_path(root, detector, tile_size)
    else:
        statistics_path = Path(statistics_path)
    time_annotations = load_time_annotations(statistics_path)
    data = load_detector_data(detector, output_root, fit_dirs)
    if reset_days is None:
        reset_days = data["reset_days"]
    reset_days = np.asarray(reset_days, dtype=float).reshape(-1)
    if reset_days.size:
        reset_day = float(reset_days[0])
    plot_data = production_plot_data(data)
    prediction_quantiles = np.stack(
        (data["modeled"], data["modeled"], data["modeled"])
    )

    save_independent_trajectory_plot(
        output_root / "numpyro_ice_trajectory.png",
        data,
        reset_day=(float(reset_days[0]) if len(reset_days) else reset_day),
        time_annotations=time_annotations,
        reset_days=reset_days,
    )
    save_rate_histogram(
        output_root / "numpyro_rate_histogram_pre_post_reset.png",
        data,
        reset_day=reset_day,
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
        time_annotations=time_annotations,
    )
    save_tile_map_plot(output_root / "numpyro_tile_ar_maps.png", data["ar_rows"])
    save_tile_rate_map_plot(
        output_root / f"numpyro_tile_rate_maps_{detector}.png",
        read_csv_rows(output_root / "tile_rate_posterior.csv"),
    )
    save_tile_fit_plot(
        output_root / "numpyro_tile_fit.png",
        plot_data,
        prediction_quantiles,
        time_annotations=time_annotations,
    )
    save_tile_data_model_time_plot(
        output_root / "numpyro_tile_data_model_vs_time.png",
        plot_data,
        prediction_quantiles,
        time_annotations=time_annotations,
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
    parser.add_argument("--reset-day", type=float, default=DEFAULT_RESET_DAY)
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=None,
        help="Use an isolated detector fit root instead of root/production/<detector>.",
    )
    parser.add_argument(
        "--statistics-path",
        type=Path,
        default=None,
        help="Statistics CSV used for event annotations with --fit-root.",
    )
    parser.add_argument(
        "--tile-size",
        type=positive_integer,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile edge length used by the production reduction; default: {DEFAULT_TILE_SIZE}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detectors = [value.upper() for value in args.detector] if args.detector else DETECTORS
    unknown = sorted(set(detectors).difference(DETECTORS))
    if unknown:
        raise ValueError(f"Unknown detectors: {unknown}")
    if (args.fit_root is not None or args.statistics_path is not None) and len(detectors) != 1:
        raise ValueError("--fit-root and --statistics-path require exactly one detector")
    for detector in detectors:
        print(f"Generating independent production-style plots for {detector}", flush=True)
        output_root = save_detector_plots(
            detector,
            args.root,
            args.reset_day,
            output_root=args.fit_root,
            statistics_path=args.statistics_path,
            tile_size=args.tile_size,
        )
        print(f"Wrote plots under {output_root}", flush=True)


if __name__ == "__main__":
    main()
