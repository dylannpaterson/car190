#!/usr/bin/env python3
"""Plot model/data and true/recovered trajectories for rate-jump tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production


LED_COLORS = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#f0e442",
    "#56b4e9",
)


def scenario_name(jump_percent):
    return f"jump_{float(jump_percent):g}pct"


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_case(output_dir, result):
    post_jump_points = int(result.get("post_jump_points", 3))
    case_dir = output_dir / f"points_{post_jump_points}" / scenario_name(
        result["jump_percent"]
    )
    if not (case_dir / "fits" / "tile_136").exists():
        case_dir = output_dir / scenario_name(result["jump_percent"])
    fit_dir = case_dir / "fits" / "tile_136"
    predictions = read_csv(fit_dir / "tile_group_predictions.csv")
    statistics = read_csv(
        case_dir / "wfi10_tile_log_statistics_256_tile_136_count_corrected.csv"
    )
    synthetic_groups = {
        int(row["group"])
        for row in statistics
        if row["file_set_name"].startswith("synthetic_rate_jump_")
    }
    with np.load(output_dir / "baseline" / "fits" / "tile_136" / "posterior_samples.npz") as archive:
        base_log_rates = np.asarray(archive["np_log_rate_nodes"], dtype=float).reshape(-1)
    with np.load(fit_dir / "posterior_samples.npz") as archive:
        recovered_rates = np.asarray(archive["node_rates"], dtype=float).reshape(-1)

    data = {
        "days": np.asarray([float(row["days"]) for row in predictions]),
        "led": np.asarray([int(row["led"]) for row in predictions]),
        "observed": np.asarray(
            [float(row["observed_log_change"]) for row in predictions]
        ),
        "model": np.asarray(
            [float(row["model_log_change"]) for row in predictions]
        ),
        "synthetic": np.asarray(
            [int(row["group"]) in synthetic_groups for row in predictions]
        ),
    }
    fit_data = production.load_tile_statistics(
        case_dir / "wfi10_tile_log_statistics_256_tile_136_count_corrected.csv"
    )
    node_days = np.asarray(fit_data["node_days"], dtype=float)
    base_rates = np.exp(base_log_rates)
    jump_fraction = float(result["jump_percent"]) / 100.0
    true_rates = np.concatenate(
        (
            base_rates,
            [base_rates[-1]]
            + [base_rates[-1] * (1.0 + jump_fraction)]
            * post_jump_points,
        )
    )
    if true_rates.shape != node_days.shape or recovered_rates.shape != node_days.shape:
        raise ValueError(
            f"Trajectory shape mismatch for {case_dir}: "
            f"days={node_days.shape}, true={true_rates.shape}, "
            f"recovered={recovered_rates.shape}"
        )
    return data, node_days, true_rates, recovered_rates, post_jump_points


def format_axes(data_axis, rate_axis, node_days, post_jump_points):
    rate_axis.axvline(
        node_days[-post_jump_points],
        color="0.25",
        linestyle="--",
        linewidth=0.9,
        label="jump node",
    )
    for axis in (data_axis, rate_axis):
        axis.grid(alpha=0.2)
        axis.set_xlim(left=0.0)


def plot_case(result, loaded, output_path):
    data, node_days, true_rates, recovered_rates, post_jump_points = loaded
    synthetic_cycle_count = post_jump_points + 1
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )
    data_axis, rate_axis = axes
    for led, color in enumerate(LED_COLORS, start=1):
        mask = data["led"] == led
        order = np.argsort(data["days"][mask])
        data_axis.scatter(
            data["days"][mask],
            1000.0 * data["observed"][mask],
            color=color,
            s=8,
            alpha=0.35,
            edgecolors="none",
        )
        data_axis.plot(
            data["days"][mask][order],
            1000.0 * data["model"][mask][order],
            color=color,
            linewidth=0.8,
            label=f"LED {led}",
        )
        synthetic_mask = mask & data["synthetic"]
        data_axis.scatter(
            data["days"][synthetic_mask],
            1000.0 * data["observed"][synthetic_mask],
            color=color,
            marker="x",
            s=28,
            linewidths=0.9,
        )

    rate_axis.plot(
        node_days,
        true_rates,
        color="black",
        linewidth=2.0,
        drawstyle="steps-post",
        label="true trajectory",
    )
    rate_axis.plot(
        node_days,
        recovered_rates,
        color="#d55e00",
        linewidth=1.5,
        label="recovered MAP trajectory",
    )
    rate_axis.scatter(
        node_days[-synthetic_cycle_count:],
        recovered_rates[-synthetic_cycle_count:],
        color="#d55e00",
        s=20,
        zorder=3,
    )
    rate_axis.scatter(
        node_days[-synthetic_cycle_count:],
        true_rates[-synthetic_cycle_count:],
        color="black",
        s=20,
        zorder=3,
    )
    format_axes(data_axis, rate_axis, node_days, post_jump_points)
    data_axis.set_ylabel("aggregate log change [10^-3]")
    data_axis.set_title("Observed synthetic data and MAP model")
    data_axis.legend(frameon=False, ncol=3, fontsize=8, loc="best")
    rate_axis.set_xlabel("Days since first exposure")
    rate_axis.set_ylabel("rate [nm/day]")
    rate_axis.set_title("True versus recovered rate trajectory")
    rate_axis.legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle(
        f"WFI10 tile 136: {result['jump_percent']:g}% jump, "
        f"{result.get('hours_post_temperature_change', 4.0 * post_jump_points):g} h "
        "post temperature change",
        y=1.02,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_grid(results, loaded_cases, output_path):
    figure, axes = plt.subplots(
        len(results),
        2,
        figsize=(14, 3.3 * len(results)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (result, loaded) in enumerate(zip(results, loaded_cases)):
        data, node_days, true_rates, recovered_rates, post_jump_points = loaded
        data_axis, rate_axis = axes[row_index]
        for led, color in enumerate(LED_COLORS, start=1):
            mask = data["led"] == led
            order = np.argsort(data["days"][mask])
            data_axis.scatter(
                data["days"][mask],
                1000.0 * data["observed"][mask],
                color=color,
                s=5,
                alpha=0.3,
                edgecolors="none",
            )
            data_axis.plot(
                data["days"][mask][order],
                1000.0 * data["model"][mask][order],
                color=color,
                linewidth=0.65,
            )
        rate_axis.plot(
            node_days,
            true_rates,
            color="black",
            linewidth=1.7,
            drawstyle="steps-post",
            label="true" if row_index == 0 else None,
        )
        rate_axis.plot(
            node_days,
            recovered_rates,
            color="#d55e00",
            linewidth=1.3,
            label="recovered" if row_index == 0 else None,
        )
        format_axes(data_axis, rate_axis, node_days, post_jump_points)
        data_axis.set_ylabel(f"{result['jump_percent']:g}%\nlog change [10^-3]")
        rate_axis.set_ylabel("nm/day")
        if row_index == len(results) - 1:
            data_axis.set_xlabel("Days since first exposure")
            rate_axis.set_xlabel("Days since first exposure")
    axes[0, 0].set_title("Observed data and MAP model")
    axes[0, 1].set_title("True and recovered trajectories")
    axes[0, 1].legend(frameon=False, fontsize=8)
    figure.suptitle("WFI10 tile 136 rate-jump sensitivity diagnostics", y=1.01)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_jump_progression(results, loaded_cases, output_path):
    ordered = sorted(
        zip(results, loaded_cases),
        key=lambda item: int(item[0]["post_jump_points"]),
    )
    figure, axes = plt.subplots(
        2,
        len(ordered),
        figsize=(5.0 * len(ordered), 8),
        sharex="row",
        constrained_layout=True,
    )
    if len(ordered) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, (result, loaded) in enumerate(ordered):
        data, node_days, true_rates, recovered_rates, post_jump_points = loaded
        data_axis, rate_axis = axes[:, column]
        for led, color in enumerate(LED_COLORS, start=1):
            mask = data["led"] == led
            order = np.argsort(data["days"][mask])
            data_axis.scatter(
                data["days"][mask],
                1000.0 * data["observed"][mask],
                color=color,
                s=4,
                alpha=0.28,
                edgecolors="none",
                label="data" if led == 1 and column == 0 else None,
            )
            data_axis.plot(
                data["days"][mask][order],
                1000.0 * data["model"][mask][order],
                color=color,
                linewidth=0.65,
                label="model" if led == 1 and column == 0 else None,
            )
        rate_axis.plot(
            node_days,
            true_rates,
            color="black",
            linewidth=1.7,
            drawstyle="steps-post",
            label="true" if column == 0 else None,
        )
        rate_axis.plot(
            node_days,
            recovered_rates,
            color="#d55e00",
            linewidth=1.3,
            label="recovered" if column == 0 else None,
        )
        format_axes(data_axis, rate_axis, node_days, post_jump_points)
        hours = result.get(
            "hours_post_temperature_change", 4.0 * post_jump_points
        )
        data_axis.set_title(f"{hours:g} h post temperature change")
        rate_axis.set_xlabel("Days")
        if column == 0:
            data_axis.set_ylabel("log change [10^-3]")
            rate_axis.set_ylabel("rate [nm/day]")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"WFI10 tile 136: {results[0]['jump_percent']:g}% jump progression",
        y=1.02,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = json.loads((args.output_dir / "rate_jump_summary.json").read_text())
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    loaded_cases = [load_case(args.output_dir, result) for result in summary["results"]]
    hours_per_point = float(summary["hours_between_synthetic_cycles"])
    for result, loaded in zip(summary["results"], loaded_cases):
        result["hours_post_temperature_change"] = (
            result["post_jump_points"] * hours_per_point
        )
        output_path = plot_dir / (
            f"points_{result['post_jump_points']}_"
            f"{scenario_name(result['jump_percent'])}_diagnostics.png"
        )
        plot_case(result, loaded, output_path)
    for jump_percent in sorted(
        {result["jump_percent"] for result in summary["results"]}
    ):
        grouped = [
            (result, loaded)
            for result, loaded in zip(summary["results"], loaded_cases)
            if result["jump_percent"] == jump_percent
        ]
        plot_jump_progression(
            [result for result, _ in grouped],
            [loaded for _, loaded in grouped],
            plot_dir / f"{scenario_name(jump_percent)}_progression.png",
        )
    for path in sorted(plot_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
