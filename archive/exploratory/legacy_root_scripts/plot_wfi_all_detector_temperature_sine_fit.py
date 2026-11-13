"""Plot independent CAR190 sinusoid fits as detector tile envelopes."""

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from analyze_wfi_post_reset_led_sinusoid import (
    DEFAULT_RESET_DAY,
    evaluate_model,
    fit_sinusoid,
    latest_block_mask,
)


DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
CAR190 = 1039
CAR216 = 1063
ACM_STEP_REPEAT = 691
DEFAULT_LED = 3
ENCLOSURE_TEMPERATURE_LABELS = {
    "WFI_MCE_ADC_WTSM10": "Encl Frame +X+Y",
    "WFI_MCE_ADC_WTSM11": "Encl Frame near CMU I/F",
    "WFI_MCE_TEMP_CMU": "CMU Internal",
}
ENCLOSURE_TEMPERATURE_BUCKET_SECONDS = 300


def parse_detectors(value):
    requested = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    unknown = sorted(set(requested) - set(DETECTORS))
    if unknown:
        raise ValueError(f"Unknown detector(s): {', '.join(unknown)}")
    if not requested:
        raise ValueError("At least one detector is required")
    return requested


def read_rows(path):
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            n_pair_valid = float(row["n_pair_valid"])
            if n_pair_valid <= 0.0:
                continue
            timestamp = datetime.fromisoformat(row["exposure_start_time"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            rows.append(
                {
                    "detector": Path(path).name[:5].upper(),
                    "tile": int(row["tile_id"]),
                    "led": int(row["led"]),
                    "program": int(row["program"]),
                    "repeat": int(row["repeat"]),
                    "days": float(row["days"]),
                    "utc": timestamp,
                    "value": float(row["sum_log_change"]) / n_pair_valid,
                }
            )
    return rows


def read_temperature_telemetry(paths, bucket_seconds=ENCLOSURE_TEMPERATURE_BUCKET_SECONDS):
    """Load long-form roman-telem CSVs and reduce them to time bins."""
    if not paths:
        return {}
    buckets = defaultdict(lambda: [0.0, 0])
    for path in paths:
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"ObsTime", "EUValue", "mnemonic"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Temperature telemetry is missing {sorted(missing)}: {path}"
                )
            for row in reader:
                mnemonic = str(row["mnemonic"]).strip()
                if mnemonic not in ENCLOSURE_TEMPERATURE_LABELS:
                    continue
                timestamp_text = str(row["ObsTime"]).strip()
                if timestamp_text.endswith("Z"):
                    timestamp_text = timestamp_text[:-1] + "+00:00"
                timestamp = datetime.fromisoformat(timestamp_text)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                timestamp = timestamp.astimezone(timezone.utc)
                try:
                    value = float(row["EUValue"])
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                bucket = int(timestamp.timestamp()) // int(bucket_seconds)
                total, count = buckets[(mnemonic, bucket)]
                buckets[(mnemonic, bucket)] = [total + value, count + 1]

    telemetry = {}
    for mnemonic in ENCLOSURE_TEMPERATURE_LABELS:
        points = []
        matching = sorted(
            (bucket, total, count)
            for (current_mnemonic, bucket), (total, count) in buckets.items()
            if current_mnemonic == mnemonic
        )
        for bucket, total, count in matching:
            points.append(
                (
                    datetime.fromtimestamp(
                        (bucket + 0.5) * int(bucket_seconds), tz=timezone.utc
                    ),
                    total / count,
                )
            )
        if points:
            telemetry[mnemonic] = points
    if not telemetry:
        raise ValueError("No recognized enclosure temperature samples were found")
    return telemetry


def acm_step_timestamp(rows):
    timestamps = [
        row["utc"] for row in rows if row["repeat"] == ACM_STEP_REPEAT
    ]
    if not timestamps:
        raise ValueError(f"No rows found for ACM step repeat {ACM_STEP_REPEAT}")
    return min(timestamps)


def fit_tile(rows, tile, led, reset_day):
    train = sorted(
        [
            row
            for row in rows
            if row["tile"] == tile
            and row["led"] == led
            and row["program"] == CAR190
            and row["days"] >= reset_day
        ],
        key=lambda row: row["days"],
    )
    holdout = sorted(
        [
            row
            for row in rows
            if row["tile"] == tile
            and row["led"] == led
            and row["program"] == CAR216
            and row["days"] >= reset_day
        ],
        key=lambda row: row["days"],
    )
    if len(train) < 6 or len(holdout) < 3:
        raise ValueError(f"Insufficient samples for tile {tile}, LED {led}")

    train_days = np.asarray([row["days"] for row in train], dtype=float)
    train_values = np.asarray([row["value"] for row in train], dtype=float)
    holdout_days = np.asarray([row["days"] for row in holdout], dtype=float)
    fit = fit_sinusoid(train_days - reset_day, train_values)
    selected_mask, _ = latest_block_mask(holdout_days)
    selected = [row for row, is_selected in zip(holdout, selected_mask) if is_selected]
    return {"train": train, "fit": fit, "selected": selected}


def envelope(results, row_key, value_function):
    grouped = defaultdict(list)
    for result in results:
        for row in result[row_key]:
            grouped[row["utc"]].append(value_function(result, row))
    times = sorted(grouped)
    values = np.asarray([grouped[time] for time in times], dtype=float)
    return times, np.min(values, axis=1), np.median(values, axis=1), np.max(values, axis=1)


def prediction_envelope(results, curve_days, reset_day):
    predictions = np.asarray(
        [evaluate_model(curve_days, result["fit"], reset_day) for result in results],
        dtype=float,
    )
    return (
        np.min(predictions, axis=0),
        np.median(predictions, axis=0),
        np.max(predictions, axis=0),
    )


def plot_fits(
    output_path,
    detector_results,
    led,
    reset_day,
    origin_utc,
    acm_step_utc,
    temperature_telemetry=None,
):
    all_results = [
        result
        for results in detector_results.values()
        for result in results
    ]
    fit_start_day = min(
        float(row["days"]) for result in all_results for row in result["train"]
    )
    selected_end_day = max(
        float(row["days"]) for result in all_results for row in result["selected"]
    )
    curve_days = np.linspace(fit_start_day, selected_end_day, 1000)
    curve_utc = [origin_utc + timedelta(days=float(day)) for day in curve_days]

    if temperature_telemetry:
        figure, (axis, temperature_axis) = plt.subplots(
            2,
            1,
            figsize=(16, 12),
            sharex=True,
            gridspec_kw={"height_ratios": (3, 1)},
            constrained_layout=True,
        )
    else:
        figure, axis = plt.subplots(figsize=(16, 10), constrained_layout=True)
        temperature_axis = None
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(detector_results)))
    detector_handles = []
    for color, (detector, results) in zip(colors, detector_results.items()):
        train_utc, train_low, train_median, train_high = envelope(
            results, "train", lambda result, row: row["value"]
        )
        selected_utc, selected_low, selected_median, selected_high = envelope(
            results, "selected", lambda result, row: row["value"]
        )
        fit_low, fit_median, fit_high = prediction_envelope(
            results, curve_days, reset_day
        )
        axis.fill_between(
            train_utc,
            1000.0 * train_low,
            1000.0 * train_high,
            color=color,
            alpha=0.12,
        )
        axis.fill_between(
            curve_utc,
            1000.0 * fit_low,
            1000.0 * fit_high,
            color=color,
            alpha=0.22,
        )
        axis.plot(
            curve_utc,
            1000.0 * fit_median,
            color=color,
            linewidth=1.25,
            alpha=0.95,
        )
        axis.fill_between(
            selected_utc,
            1000.0 * selected_low,
            1000.0 * selected_high,
            color=color,
            alpha=0.42,
        )
        detector_handles.append(
            Line2D([], [], color=color, linewidth=2.0, label=detector)
        )

    if temperature_telemetry:
        temperature_handles = []
        temperature_colors = plt.get_cmap("Set2")(
            np.linspace(0.0, 1.0, len(temperature_telemetry))
        )
        for color, (mnemonic, points) in zip(
            temperature_colors, temperature_telemetry.items()
        ):
            times = [point[0] for point in points]
            values = [point[1] for point in points]
            temperature_axis.plot(
                times,
                values,
                color=color,
                linewidth=0.8,
                alpha=0.85,
                label=ENCLOSURE_TEMPERATURE_LABELS[mnemonic],
            )
            temperature_handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    linewidth=1.5,
                    label=ENCLOSURE_TEMPERATURE_LABELS[mnemonic],
                )
            )
        temperature_axis.axvline(
            acm_step_utc,
            color="black",
            linestyle="--",
            linewidth=1.25,
        )
        temperature_axis.set_ylabel("Enclosure telemetry temperature (K)")
        temperature_axis.set_title("Roman MAST enclosure telemetry", fontsize=11)
        temperature_axis.grid(alpha=0.2)
        temperature_axis.legend(
            handles=temperature_handles,
            title="Roman MAST telemetry",
            frameon=False,
            loc="upper right",
            fontsize=8,
        )

    axis.axvline(
        acm_step_utc,
        color="black",
        linestyle="--",
        linewidth=1.25,
    )
    axis.set_ylabel("Mean log change (millilog)")
    axis.set_title(f"WFI LED {led}: detector tile envelopes, CAR190 and CAR216")
    time_axis = temperature_axis if temperature_axis is not None else axis
    time_axis.set_xlabel("Exposure start time (UTC)")
    time_axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc)
    )
    time_axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axis.set_xlim(curve_utc[0], curve_utc[-1])
    axis.grid(alpha=0.2)
    detector_legend = axis.legend(
        handles=detector_handles,
        title="detector color",
        frameon=False,
        ncol=3,
        loc="upper left",
        fontsize=9,
    )
    axis.add_artist(detector_legend)
    axis.legend(
        handles=[
            Patch(facecolor="0.35", alpha=0.12, label="CAR190 tile envelope"),
            Line2D([], [], color="0.35", linewidth=1.4, label="CAR190 sine median"),
            Patch(facecolor="0.35", alpha=0.42, label="CAR216 tile envelope"),
            Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=1.25,
                label=(
                    f"ACM step: repeat {ACM_STEP_REPEAT}, "
                    f"{acm_step_utc.strftime('%Y-%m-%d %H:%M UTC')}"
                ),
            ),
        ],
        frameon=False,
        loc="lower left",
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument(
        "--detectors",
        type=parse_detectors,
        default=DETECTORS,
        help="Comma-separated detector list (default: all WFI detectors)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--led", type=int, default=DEFAULT_LED)
    parser.add_argument("--reset-day", type=float, default=DEFAULT_RESET_DAY)
    parser.add_argument(
        "--enclosure-temperature-csv",
        type=Path,
        action="append",
        help=(
            "Long-form roman-telem CSV; repeat for the three temp_enclosure "
            "mnemonics to add a five-minute temperature overlay."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    temperature_telemetry = read_temperature_telemetry(
        args.enclosure_temperature_csv
    )
    detector_results = {}
    all_rows = []
    for detector in args.detectors:
        statistics_path = args.statistics_dir / (
            f"{detector.lower()}_tile_log_statistics_512_count_corrected.csv"
        )
        rows = read_rows(statistics_path)
        all_rows.extend(rows)
        tiles = sorted({row["tile"] for row in rows})
        detector_results[detector] = [
            fit_tile(rows, tile, args.led, args.reset_day) for tile in tiles
        ]

    origin_utc = min(
        row["utc"] - timedelta(days=float(row["days"])) for row in all_rows
    )
    acm_step_utc = acm_step_timestamp(all_rows)
    plot_fits(
        args.output,
        detector_results,
        args.led,
        args.reset_day,
        origin_utc,
        acm_step_utc,
        temperature_telemetry=temperature_telemetry,
    )
    tile_count = sum(len(results) for results in detector_results.values())
    print(f"Detectors: {', '.join(detector_results)}")
    print(f"Independent fits: {tile_count} tiles, LED {args.led}")
    print(
        "Tiles per detector: "
        + ", ".join(
            f"{detector}={len(results)}"
            for detector, results in detector_results.items()
        )
    )
    print("Fit input: CAR190 only")
    print("Displayed comparison envelope: CAR216")
    print(f"ACM step repeat {ACM_STEP_REPEAT}: {acm_step_utc.isoformat()}")
    if temperature_telemetry:
        print(
            "Enclosure telemetry: "
            + ", ".join(
                f"{ENCLOSURE_TEMPERATURE_LABELS[mnemonic]}={len(points)} bins"
                for mnemonic, points in temperature_telemetry.items()
            )
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
