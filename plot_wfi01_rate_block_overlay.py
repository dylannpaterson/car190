"""Overlay WFI01 production rates on the supplied reference rate blocks."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from wfi_production_layout import DEFAULT_TILE_SIZE, positive_integer
from wfi_production_plot_annotations import load_time_annotations


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "20261014" / "corrected_statistics_4h_thinned"
DEFAULT_TELEMETRY = (
    ROOT
    / "20261011"
    / "reduction_stage_comparison"
    / "clean_program1039_tmm_wfi01_tile000"
    / "telemetry"
    / "wfi_mce_adc_wtsm00.csv"
)
DEFAULT_RESET_DAY = 12.907397928240743

# Intervals and uncertainty envelopes are transcribed from the supplied figure.
REFERENCE_BLOCKS = (
    ("decon1", 4.0, 20.0, 12.61, 12.08, 13.20),
    ("decon1", 18.0, 70.0, 11.88, 10.90, 12.48),
    ("decon1", 70.0, 270.0, 10.24, 9.45, 10.96),
    ("decon2", 310.0, 325.0, 17.27, 15.85, 18.18),
    ("decon2", 325.0, 340.0, 10.99, 10.10, 11.52),
    ("decon2", 340.0, 405.0, 12.10, 11.10, 12.70),
    ("decon2", 405.0, 500.0, 10.26, 9.40, 10.98),
    ("decon2", 500.0, 600.0, 9.46, 8.55, 10.00),
    ("decon2", 600.0, 765.0, 9.21, 8.45, 9.70),
)
REFERENCE_MULTIPLIERS = (1.232, 1.160, 1.000, 1.687, 1.073, 1.181, 1.002, 0.924, 0.899)


def parse_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_production_rates(fit_root):
    paths = sorted(Path(fit_root).glob("tile_*/posterior_samples.npz"))
    if len(paths) != 256:
        raise ValueError(f"Expected 256 WFI01 tile archives, found {len(paths)}")
    days = None
    rates = []
    for path in paths:
        with np.load(path, allow_pickle=False) as samples:
            tile_days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            tile_rates = np.asarray(
                samples["tile_instantaneous_rate_nm_day"], dtype=float
            ).reshape(-1)
            if days is None:
                days = tile_days
            elif not np.array_equal(days, tile_days):
                raise ValueError(f"Observation days differ in {path}")
            if tile_rates.size != days.size:
                raise ValueError(f"Rate trajectory length differs in {path}")
            rates.append(tile_rates)
    rates = np.asarray(rates, dtype=float)
    return days * 24.0, np.nanpercentile(rates, [16, 50, 84], axis=0)


def load_temperature(path, start_time, maximum_hours):
    hours = []
    values = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = parse_timestamp(row["ObsTime"])
            value = float(row["EUValue"])
            elapsed = (timestamp - start_time).total_seconds() / 3600.0
            if 0.0 <= elapsed <= maximum_hours and np.isfinite(value):
                hours.append(elapsed)
                values.append(value)
    return np.asarray(hours), np.asarray(values)


def load_start_time(statistics_path):
    with Path(statistics_path).open(newline="") as handle:
        timestamps = [
            parse_timestamp(row["exposure_start_time"])
            for row in csv.DictReader(handle)
        ]
    if not timestamps:
        raise ValueError(f"No exposure timestamps found in {statistics_path}")
    return min(timestamps)


def draw_reference_blocks(axis):
    colors = {"decon1": "#377eb8", "decon2": "#e66a3a"}
    for kind, start, stop, rate, lower, upper in REFERENCE_BLOCKS:
        color = colors[kind]
        axis.fill_between(
            (start, stop),
            (lower, lower),
            (upper, upper),
            color=color,
            alpha=0.17,
            linewidth=0.0,
        )
        axis.hlines(y=rate, xmin=start, xmax=stop, color=color, linewidth=2.7)
        axis.text(
            (start + stop) / 2.0,
            rate + 0.16,
            f"{rate:.2f}",
            color="#555555",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_plot(data_root, statistics_path, telemetry_path, output, reset_day, tile_size):
    fit_root = (
        Path(data_root)
        / "production"
        / f"tile_size_{tile_size}"
        / "wfi01"
        / "fits"
    )
    hours, rate_quantiles = load_production_rates(fit_root)
    start_time = load_start_time(statistics_path)
    telemetry_hours, telemetry_values = load_temperature(
        telemetry_path,
        start_time,
        maximum_hours=max(800.0, float(hours[-1])),
    )
    annotations = load_time_annotations(statistics_path)

    figure, (rate_axis, temperature_axis) = plt.subplots(
        2,
        1,
        figsize=(13.2, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 0.82)},
    )
    draw_reference_blocks(rate_axis)
    rate_axis.fill_between(
        hours,
        rate_quantiles[0],
        rate_quantiles[2],
        color="#009e73",
        alpha=0.18,
        linewidth=0.0,
        label="WFI01 production (16-84% across 256 tiles)",
    )
    rate_axis.plot(
        hours,
        rate_quantiles[1],
        color="#007f5f",
        linewidth=1.4,
        label="WFI01 production median",
    )

    reset_hours = reset_day * 24.0
    rate_axis.axvline(
        reset_hours,
        color="#555555",
        linewidth=1.1,
        label="decon reset",
    )
    temperature_axis.axvline(reset_hours, color="#555555", linewidth=1.1)
    event_labels = set()
    for event in annotations.events:
        if event.kind == "side_swap":
            color, linestyle, label = "#7b3294", "-.", "MCE side change"
        elif event.kind == "acm":
            color, linestyle, label = "#777777", "--", "ACM move"
        else:
            continue
        event_hours = event.day * 24.0
        rate_axis.axvline(
            event_hours,
            color=color,
            linestyle=linestyle,
            linewidth=0.8,
            alpha=0.85,
            label=label if label not in event_labels else None,
        )
        temperature_axis.axvline(
            event_hours,
            color=color,
            linestyle=linestyle,
            linewidth=0.7,
            alpha=0.75,
        )
        event_labels.add(label)

    temperature_axis.plot(
        telemetry_hours,
        telemetry_values,
        color="#555555",
        linewidth=1.2,
        label="MPA temperature (WTSM00)",
    )
    rate_axis.set_xlim(0.0, 800.0)
    rate_axis.set_ylim(7.0, 18.8)
    temperature_axis.set_ylim(88.0, 126.0)
    rate_axis.set_ylabel("deposition rate (nm/day)")
    temperature_axis.set_ylabel("MPA temperature (K)")
    temperature_axis.set_xlabel("elapsed time (h)")
    rate_axis.grid(alpha=0.2)
    temperature_axis.grid(alpha=0.2)
    rate_axis.set_title("WFI01 production rate overlaid on reference deposition blocks")
    figure.suptitle(
        "Reference: N = 64; WFI01 production: N = 256\n"
        "Reference multipliers: "
        + ", ".join(f"{value:.3f}" for value in REFERENCE_MULTIPLIERS),
        y=0.99,
        fontsize=12,
    )
    handles = [
        Line2D([], [], color="#377eb8", linewidth=2.7, label="decon1 blocks (reference)"),
        Line2D([], [], color="#e66a3a", linewidth=2.7, label="decon2 blocks (reference)"),
        Line2D([], [], color="#007f5f", linewidth=1.4, label="WFI01 production median"),
        Line2D([], [], color="#555555", linewidth=1.1, label="decon reset"),
        Line2D([], [], color="#777777", linestyle="--", linewidth=0.8, label="ACM move"),
        Line2D([], [], color="#7b3294", linestyle="-.", linewidth=0.8, label="MCE side change"),
    ]
    rate_axis.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    temperature_axis.legend(loc="upper right", frameon=False, fontsize=8)
    figure.subplots_adjust(top=0.86, bottom=0.09, hspace=0.10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Wrote {output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--statistics-path", type=Path, default=None)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reset-day", type=float, default=DEFAULT_RESET_DAY)
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    statistics_path = args.statistics_path or (
        args.data_root
        / f"wfi01_tile_log_statistics_{args.tile_size}_count_corrected.csv"
    )
    output = args.output or (
        args.data_root
        / "production"
        / f"tile_size_{args.tile_size}"
        / "wfi01"
        / "wfi01_rate_block_overlay.png"
    )
    make_plot(
        args.data_root,
        statistics_path,
        args.telemetry,
        output,
        args.reset_day,
        args.tile_size,
    )


if __name__ == "__main__":
    main()
