"""Download and plot flight sRCS photodiode telemetry for local RCS flats."""

import argparse
import ast
import bisect
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_URI = "https://mast.stsci.edu/edp/api/v0.1/mnemonics/spa/roman/data"
PHOTODIODE_MNEMONICS = (
    "WFI_MCE_SRCS_PD1_V",
    "WFI_MCE_SRCS_PD2_V",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "all_wfi_rcs_flat_index_20260930.csv",
        help="Complete local flat index used to define exposure windows",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=ROOT / "mast_api_token.txt",
        help="File containing the MAST API token",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "rrn_reduction_20260930"
        / "wfi_rcs_photodiode_telemetry_20260912_20260930",
        help="Directory for downloaded telemetry, summaries, and plots",
    )
    parser.add_argument(
        "--statistics-root",
        type=Path,
        default=ROOT / "rrn_reduction_20260930",
        help="Directory containing the RRN tile statistics used for detector data",
    )
    return parser.parse_args()


def _parse_time(value):
    return datetime.fromisoformat(str(value).rstrip("Z"))


def _read_token(path):
    token = path.read_text().strip()
    if token.startswith("MAST_API_TOKEN="):
        token = token.split("=", 1)[1].strip()
    if not token:
        raise ValueError(f"MAST token file is empty: {path}")
    return token


def _load_exposures(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No flat metadata rows found in {path}")

    groups = {}
    for row in rows:
        start = _parse_time(row["exposure_start_time"])
        end = _parse_time(row["exposure_end_time"])
        key = (start, end, int(row["led"]), int(row["bank"]))
        group = groups.setdefault(
            key,
            {
                "start": start,
                "end": end,
                "led": int(row["led"]),
                "bank": int(row["bank"]),
                "rows": [],
            },
        )
        group["rows"].append(row)

    exposures = sorted(groups.values(), key=lambda item: item["start"])
    detector_counts = {len(exposure["rows"]) for exposure in exposures}
    if len(detector_counts) != 1:
        raise ValueError(
            "Flat metadata does not contain the same number of detectors per exposure: "
            f"{sorted(detector_counts)}"
        )
    return exposures


def _query_mnemonic(token, mnemonic, start, end):
    query = urlencode(
        {
            "mnemonic": mnemonic,
            "s_time": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "e_time": end.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    request = Request(
        f"{DATA_URI}?{query}",
        headers={"Authorization": f"token {token}", "Accept": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        raw = response.read().decode()
    payload = json.loads(raw) if raw.lstrip().startswith(("{", "[")) else ast.literal_eval(raw)
    values = []
    for item in payload.get("Data", []):
        value = item.get("EUValue")
        if isinstance(value, (int, float)):
            values.append((_parse_time(item["ObsTime"]), float(value)))
    if not values:
        raise ValueError(f"No numeric telemetry returned for {mnemonic}")
    return sorted(values)


def _values_in_window(values, start, end):
    times = [item[0] for item in values]
    lower = bisect.bisect_left(times, start)
    upper = bisect.bisect_right(times, end)
    selected = [value for _, value in values[lower:upper]]
    if not selected:
        raise ValueError(
            f"No telemetry samples overlap {start.isoformat()} to {end.isoformat()}"
        )
    return np.asarray(selected, dtype=float)


def _days_since(start, value):
    return (value - start).total_seconds() / 86400.0


def _load_detector_statistics(path):
    paths = sorted(path.glob("wfi??_tile_log_statistics_512.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No RRN detector statistics found in {path}: "
            "expected wfi??_tile_log_statistics_512.csv"
        )

    detector_statistics = {}
    reference_keys = None
    for statistics_path in paths:
        detector = statistics_path.name.split("_", 1)[0].upper()
        grouped = {}
        with statistics_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"led", "exposure_start_time", "n_pair_valid", "sum_current_counts"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{statistics_path} is missing detector count fields: {sorted(missing)}"
                )
            for row in reader:
                key = (_parse_time(row["exposure_start_time"]), int(row["led"]))
                total, count = grouped.setdefault(key, [0.0, 0])
                grouped[key][0] = total + float(row["sum_current_counts"])
                grouped[key][1] = count + int(row["n_pair_valid"])
        values = {
            key: total / count
            for key, (total, count) in grouped.items()
            if count > 0
        }
        if len(values) != len(grouped):
            raise ValueError(f"{statistics_path} contains an empty detector group")
        keys = set(values)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(
                f"RRN detector statistics do not have matching exposure groups: {statistics_path}"
            )
        detector_statistics[detector] = values
    return detector_statistics


def _detector_reference_values(detector_statistics):
    references = {}
    for detector, values in detector_statistics.items():
        references[detector] = {}
        for led in range(1, 7):
            candidates = [
                (start, value)
                for (start, row_led), value in values.items()
                if row_led == led
            ]
            if not candidates:
                raise ValueError(f"RRN detector statistics have no LED {led}: {detector}")
            references[detector][led] = min(candidates)[1]
    return references


def _write_raw(path, telemetry):
    times = sorted({time for values in telemetry.values() for time, _ in values})
    values_by_time = {
        name: {time: value for time, value in values}
        for name, values in telemetry.items()
    }
    fields = ["obs_time", *telemetry]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for time in times:
            writer.writerow(
                {
                    "obs_time": time.isoformat(timespec="milliseconds"),
                    **{
                        name: values_by_time[name].get(time, "")
                        for name in telemetry
                    },
                }
            )


def _summarize_exposures(
    exposures,
    telemetry,
    mission_start,
    detector_statistics,
    detector_references,
):
    summaries = []
    for exposure in exposures:
        detector_values = np.asarray(
            [
                detector_statistics[detector][(exposure["start"], exposure["led"])]
                for detector in sorted(detector_statistics)
            ],
            dtype=float,
        )
        detector_relative = np.asarray(
            [
                100.0
                * (
                    detector_values[index]
                    / detector_references[detector][exposure["led"]]
                    - 1.0
                )
                for index, detector in enumerate(sorted(detector_statistics))
            ],
            dtype=float,
        )
        summary = {
            "exposure_start_time": exposure["start"].isoformat(timespec="milliseconds"),
            "exposure_end_time": exposure["end"].isoformat(timespec="milliseconds"),
            "time_days": _days_since(mission_start, exposure["start"]),
            "led": exposure["led"],
            "bank": exposure["bank"],
            "n_archive_detectors": len(exposure["rows"]),
            "n_statistics_detectors": len(detector_statistics),
            "archive_image_median_mean": float(
                np.mean([float(row["image_median"]) for row in exposure["rows"]])
            ),
            "archive_image_median_median": float(
                np.median([float(row["image_median"]) for row in exposure["rows"]])
            ),
            "detector_count_mean": float(np.mean(detector_values)),
            "detector_count_median": float(np.median(detector_values)),
            "detector_relative_median_percent": float(np.median(detector_relative)),
        }
        for name, values in telemetry.items():
            selected = _values_in_window(values, exposure["start"], exposure["end"])
            short_name = name.removeprefix("WFI_MCE_SRCS_").lower()
            summary[f"{short_name}_mean"] = float(np.mean(selected))
            summary[f"{short_name}_median"] = float(np.median(selected))
            summary[f"{short_name}_n_samples"] = int(selected.size)
        summaries.append(summary)
    return summaries


def _write_summaries(path, summaries):
    fields = list(summaries[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def _normalized(summaries, field):
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[int(summary["led"])].append(summary)
    result = {}
    for led, values in grouped.items():
        values.sort(key=lambda item: item["exposure_start_time"])
        reference = float(values[0][field])
        result[led] = (
            values,
            np.asarray(
                [100.0 * (float(value[field]) / reference - 1.0) for value in values]
            ),
        )
    return result


def _plot_raw(output_path, telemetry, summaries, mission_start):
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for axis, name in zip(axes, PHOTODIODE_MNEMONICS):
        values = telemetry[name]
        times = [_days_since(mission_start, time) for time, _ in values]
        measurements = [value for _, value in values]
        axis.plot(times, measurements, linewidth=0.45, alpha=0.75, label=name)
        centers = [float(item["time_days"]) for item in summaries]
        means = [float(item[f"{name.removeprefix('WFI_MCE_SRCS_').lower()}_mean"]) for item in summaries]
        axis.scatter(
            centers,
            means,
            c=[colors((int(item["led"]) - 1) % 10) for item in summaries],
            s=14,
            zorder=3,
            label="flat exposure means",
        )
        axis.set_ylabel("Voltage (V)")
        axis.set_title(name)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[-1].set_xlabel("Days since mission start (first WFI exposure)")
    figure.suptitle(
        "Flight sRCS photodiode telemetry with matched WFI flat exposures\n"
        f"Mission start: {mission_start.isoformat(timespec='milliseconds')} UTC"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_normalized(output_path, summaries):
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fields = (
        "pd1_v_mean",
        "pd2_v_mean",
    )
    for axis, field in zip(axes, fields):
        normalized = _normalized(summaries, field)
        for led, (values, changes) in sorted(normalized.items()):
            axis.plot(
                [float(value["time_days"]) for value in values],
                changes,
                marker="o",
                linewidth=1.2,
                label=f"Archive LED {led}",
            )
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_ylabel("Relative output (%)")
        axis.set_title(field.replace("_", " ").upper())
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Days since mission start (first WFI exposure)")
    figure.suptitle("Per-LED photodiode output normalized to the first exposure")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_image_comparison(output_path, summaries):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    normalized_fields = (
        "pd1_v_mean",
        "pd2_v_mean",
        "detector_relative_median_percent",
    )
    for led, axis in zip(range(1, 7), axes.flat):
        values = sorted(
            [summary for summary in summaries if int(summary["led"]) == led],
            key=lambda item: item["exposure_start_time"],
        )
        for field, color, label in zip(
            normalized_fields,
            ("tab:blue", "tab:orange", "black"),
            ("PD1", "PD2", "RRN detector-count median"),
        ):
            if field == "detector_relative_median_percent":
                changes = [float(value[field]) for value in values]
            else:
                reference = float(values[0][field])
                changes = [
                    100.0 * (float(value[field]) / reference - 1.0)
                    for value in values
                ]
            axis.plot(
                [float(value["time_days"]) for value in values],
                changes,
                marker="o",
                linewidth=1.2,
                color=color,
                label=label,
            )
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.set_title(f"Archive LED {led}")
        axis.set_xlabel("Days since mission start")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Relative change (%)")
    axes[1, 0].set_ylabel("Relative change (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Photodiode output versus RRN detector-count statistics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    exposures = _load_exposures(args.metadata)
    mission_start = min(item["start"] for item in exposures)
    statistics_root = (
        args.statistics_root
        if args.statistics_root.is_absolute()
        else ROOT / args.statistics_root
    )
    detector_statistics = _load_detector_statistics(statistics_root)
    detector_references = _detector_reference_values(detector_statistics)
    token = _read_token(args.token_file)
    query_start = min(item["start"] for item in exposures) - timedelta(minutes=1)
    query_end = max(item["end"] for item in exposures) + timedelta(minutes=1)
    telemetry = {
        name: _query_mnemonic(token, name, query_start, query_end)
        for name in PHOTODIODE_MNEMONICS
    }
    summaries = _summarize_exposures(
        exposures,
        telemetry,
        mission_start,
        detector_statistics,
        detector_references,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_raw(args.output_dir / "flight_photodiode_telemetry.csv", telemetry)
    _write_summaries(args.output_dir / "flat_exposure_photodiode_summary.csv", summaries)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": str(args.metadata.resolve()),
                "mnemonics": list(PHOTODIODE_MNEMONICS),
                "query_start": query_start.isoformat(),
                "query_end": query_end.isoformat(),
                "mission_start": mission_start.isoformat(),
                "mission_start_definition": "first WFI flat exposure start time",
                "detector_statistics_root": str(statistics_root.resolve()),
                "detector_statistics_detectors": sorted(detector_statistics),
                "detector_statistics_definition": (
                    "Per-detector valid-pixel weighted mean of sum_current_counts; "
                    "comparison curve is the median relative change across available RRN detectors"
                ),
                "n_exposures": len(exposures),
                "n_telemetry_samples": {name: len(values) for name, values in telemetry.items()},
            },
            indent=2,
        )
        + "\n"
    )
    _plot_raw(args.output_dir / "flight_photodiode_timeseries.png", telemetry, summaries, mission_start)
    _plot_normalized(args.output_dir / "flight_photodiode_per_led_normalized.png", summaries)
    _plot_image_comparison(args.output_dir / "flight_photodiode_image_comparison.png", summaries)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "n_exposures": len(exposures), "n_telemetry_samples": {name: len(values) for name, values in telemetry.items()}}, indent=2))


if __name__ == "__main__":
    main()
