"""Plot local count-corrected WFI tile means against Eric's calibrated means.

The comparison is keyed by the product identity encoded in the local file-set
name and Eric run label: program, decon cycle, pass, and segment.  LED3 has
several local observations within a product key; the observation nearest Eric's
product timestamp is selected for that LED.  The extra local 1039 pass 025 is
reported and excluded when it is not present in Eric's product set.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
DEFAULT_DETECTORS = ("WFI01", "WFI10")
DEFAULT_TILES = (136, 137)
DEFAULT_LEDS = tuple(range(1, 7))

ERIC_RUN_RE = re.compile(
    r"^(?:CAR\d+_)?(?P<program>\d{5})_(?P<decon>\d{2})_"
    r"(?P<pass_number>\d{3})_(?P<segment>\d{3})(?:_|$)"
)
LOCAL_RUN_RE = re.compile(
    r"^r(?P<program>\d{5})(?P<decon>\d{2})"
    r"(?P<pass_number>\d{3})(?P<segment>\d{3})"
)


def parse_timestamp(value):
    """Parse the seven-digit fractional seconds used by the CSV files."""
    return datetime.fromisoformat(value.rstrip("0") if value.endswith("0000") else value)


def product_key(value, pattern):
    match = pattern.match(value)
    if match is None:
        raise ValueError(f"Could not parse product identity: {value}")
    return tuple(match.group(name) for name in ("program", "decon", "pass_number", "segment"))


def read_local_rows(path, tiles):
    rows = defaultdict(list)
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            tile_id = int(row["tile_id"])
            if tile_id not in tiles:
                continue
            key = (
                product_key(row["file_set_name"], LOCAL_RUN_RE),
                int(row["led"]),
                tile_id,
            )
            rows[key].append(row)
    if not rows:
        raise ValueError(f"No requested local rows found in {path}")
    return rows


def read_eric_rows(path, detectors, tiles):
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["detector"] not in detectors or int(row["tile_id"]) not in tiles:
                continue
            row["product_key"] = product_key(row["run_label"], ERIC_RUN_RE)
            rows.append(row)
    if not rows:
        raise ValueError(f"No requested Eric rows found in {path}")
    return rows


def mean_counts(row):
    n_valid = float(row["n_pair_valid"])
    if n_valid <= 0.0:
        raise ValueError(f"Non-positive valid-pair count in {row}")
    return float(row["sum_current_counts"]) / n_valid


def build_comparison(eric_rows, local_rows, detectors, tiles):
    eric_product_keys = {(row["detector"], row["product_key"]) for row in eric_rows}
    local_product_keys = {
        (detector, key)
        for detector in detectors
        for key, _, _ in local_rows[detector].keys()
    }
    extra_local_keys = {
        detector: sorted(
            key for det, key in local_product_keys - eric_product_keys if det == detector
        )
        for detector in detectors
    }

    rows = []
    for eric in eric_rows:
        detector = eric["detector"]
        led = int(eric["led"])
        tile_id = int(eric["tile_id"])
        key = (eric["product_key"], led, tile_id)
        candidates = local_rows[detector].get(key, [])
        if not candidates:
            raise ValueError(
                "Missing local corrected row for "
                f"{detector}, {eric['run_label']}, LED {led}, tile {tile_id}"
            )
        eric_time = parse_timestamp(eric["timestamp"])
        local = min(
            candidates,
            key=lambda row: abs(parse_timestamp(row["exposure_start_time"]) - eric_time),
        )
        local_time = parse_timestamp(local["exposure_start_time"])
        rows.append(
            {
                "detector": detector,
                "product_key": "_".join(eric["product_key"]),
                "run_label": eric["run_label"],
                "timestamp": eric["timestamp"],
                "days": 0.0,
                "led": led,
                "tile_id": tile_id,
                "local_file_set_name": local["file_set_name"],
                "time_offset_seconds": (local_time - eric_time).total_seconds(),
                "ours_corrected_mean": mean_counts(local),
                "eric_calibrated_mean": float(eric["calibrated_mean_dn_s"]),
            }
        )

    for detector in detectors:
        detector_rows = [row for row in rows if row["detector"] == detector]
        first_time = min(parse_timestamp(row["timestamp"]) for row in detector_rows)
        for row in detector_rows:
            row["days"] = (parse_timestamp(row["timestamp"]) - first_time).total_seconds() / 86400.0

        for led in DEFAULT_LEDS:
            for tile_id in tiles:
                series = sorted(
                    (
                        row
                        for row in detector_rows
                        if row["led"] == led and row["tile_id"] == tile_id
                    ),
                    key=lambda row: row["timestamp"],
                )
                if not series:
                    raise ValueError(f"No comparison rows for {detector}, LED {led}, tile {tile_id}")
                ours_first = series[0]["ours_corrected_mean"]
                eric_first = series[0]["eric_calibrated_mean"]
                for row in series:
                    row["ours_normalized_percent"] = 100.0 * (
                        row["ours_corrected_mean"] / ours_first - 1.0
                    )
                    row["eric_normalized_percent"] = 100.0 * (
                        row["eric_calibrated_mean"] / eric_first - 1.0
                    )
                    row["residual_percent"] = 100.0 * (
                        row["ours_corrected_mean"] / row["eric_calibrated_mean"] - 1.0
                    )

    return rows, extra_local_keys


def _series(rows, detector, led, tile_id):
    return sorted(
        (
            row
            for row in rows
            if row["detector"] == detector and row["led"] == led and row["tile_id"] == tile_id
        ),
        key=lambda row: row["days"],
    )


def _configure_axes(axes, title, ylabel):
    for axis, led in zip(axes.flat, DEFAULT_LEDS):
        axis.set_title(f"LED {led}")
        axis.set_xlabel("Days since first Eric product")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].figure.suptitle(title, y=0.985)


def plot_absolute(output, rows, detector, tiles):
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    colors = {"ours": "#d95f02", "eric": "#1b9e77"}
    styles = {136: "-", 137: "--"}
    for axis, led in zip(axes.flat, DEFAULT_LEDS):
        for tile_id in tiles:
            series = _series(rows, detector, led, tile_id)
            x = [row["days"] for row in series]
            axis.plot(
                x,
                [row["ours_corrected_mean"] for row in series],
                color=colors["ours"],
                linestyle=styles[tile_id],
                linewidth=1.0,
            )
            axis.plot(
                x,
                [row["eric_calibrated_mean"] for row in series],
                color=colors["eric"],
                linestyle=styles[tile_id],
                linewidth=1.0,
            )
    _configure_axes(axes, f"{detector}: local corrected counts versus Eric calibrated counts", "Mean counts / s")
    handles = [
        Line2D([0], [0], color=colors["ours"], linewidth=1.5, label="Ours: count-corrected"),
        Line2D([0], [0], color=colors["eric"], linewidth=1.5, label="Eric: calibrated"),
        Line2D([0], [0], color="0.25", linestyle="-", label="Tile 136"),
        Line2D([0], [0], color="0.25", linestyle="--", label="Tile 137"),
    ]
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.945))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_normalized(output, rows, detector, tiles):
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    colors = {"ours": "#d95f02", "eric": "#1b9e77"}
    styles = {136: "-", 137: "--"}
    for axis, led in zip(axes.flat, DEFAULT_LEDS):
        for tile_id in tiles:
            series = _series(rows, detector, led, tile_id)
            x = [row["days"] for row in series]
            axis.plot(
                x,
                [row["ours_normalized_percent"] for row in series],
                color=colors["ours"],
                linestyle=styles[tile_id],
                linewidth=1.0,
            )
            axis.plot(
                x,
                [row["eric_normalized_percent"] for row in series],
                color=colors["eric"],
                linestyle=styles[tile_id],
                linewidth=1.0,
            )
        axis.axhline(0.0, color="0.25", linewidth=0.7)
    _configure_axes(axes, f"{detector}: normalized corrected-count trends", "Change from first product (%)")
    handles = [
        Line2D([0], [0], color=colors["ours"], linewidth=1.5, label="Ours: count-corrected"),
        Line2D([0], [0], color=colors["eric"], linewidth=1.5, label="Eric: calibrated"),
        Line2D([0], [0], color="0.25", linestyle="-", label="Tile 136"),
        Line2D([0], [0], color="0.25", linestyle="--", label="Tile 137"),
    ]
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.945))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_residual(output, rows, detector, tiles):
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    colors = {136: "#7570b3", 137: "#e7298a"}
    styles = {136: "-", 137: "--"}
    for axis, led in zip(axes.flat, DEFAULT_LEDS):
        for tile_id in tiles:
            series = _series(rows, detector, led, tile_id)
            axis.plot(
                [row["days"] for row in series],
                [row["residual_percent"] for row in series],
                color=colors[tile_id],
                linestyle=styles[tile_id],
                linewidth=1.0,
            )
        axis.axhline(0.0, color="0.25", linewidth=0.7)
    _configure_axes(axes, f"{detector}: local corrected / Eric calibrated residual", "Ours / Eric - 1 (%)")
    handles = [
        Line2D([0], [0], color=colors[136], linewidth=1.5, label="Tile 136"),
        Line2D([0], [0], color=colors[137], linestyle="--", linewidth=1.5, label="Tile 137"),
    ]
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.945))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_csv(path, rows):
    fields = (
        "detector",
        "product_key",
        "run_label",
        "timestamp",
        "days",
        "led",
        "tile_id",
        "local_file_set_name",
        "time_offset_seconds",
        "ours_corrected_mean",
        "eric_calibrated_mean",
        "ours_normalized_percent",
        "eric_normalized_percent",
        "residual_percent",
    )
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eric-tile-trends", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eric_rows = read_eric_rows(args.eric_tile_trends, DEFAULT_DETECTORS, DEFAULT_TILES)
    local_rows = {
        detector: read_local_rows(
            args.local_root / "corrected_statistics" / f"{detector.lower()}_tile_log_statistics_256_count_corrected.csv",
            DEFAULT_TILES,
        )
        for detector in DEFAULT_DETECTORS
    }
    rows, extra_local_keys = build_comparison(eric_rows, local_rows, DEFAULT_DETECTORS, DEFAULT_TILES)

    write_csv(args.output_dir / "eric_local_corrected_counts_matched.csv", rows)
    for detector in DEFAULT_DETECTORS:
        plot_absolute(args.output_dir / f"{detector.lower()}_corrected_counts_absolute.png", rows, detector, DEFAULT_TILES)
        plot_normalized(args.output_dir / f"{detector.lower()}_corrected_counts_normalized.png", rows, detector, DEFAULT_TILES)
        plot_residual(args.output_dir / f"{detector.lower()}_corrected_counts_residual.png", rows, detector, DEFAULT_TILES)

    print(f"matched_rows={len(rows)}")
    for detector in DEFAULT_DETECTORS:
        print(f"{detector}_extra_local_product_keys={extra_local_keys[detector]}")
    print(f"wrote={args.output_dir}")


if __name__ == "__main__":
    main()
