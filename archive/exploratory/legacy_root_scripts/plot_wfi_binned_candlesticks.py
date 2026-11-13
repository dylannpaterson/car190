import csv
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import asdf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "wfi01_rcs_flat_index.csv"
DATA_DIR = ROOT / "MAST/Roman"
BIN_SIZE = 100

AXES = {
    "x": {
        "dimension": 1,
        "label": "X pixel",
        "csv_path": ROOT / "wfi01_x_binned_candlesticks.csv",
        "plot_path": ROOT / "wfi01_x_binned_candlesticks.png",
    },
    "y": {
        "dimension": 0,
        "label": "Y pixel",
        "csv_path": ROOT / "wfi01_y_binned_candlesticks.csv",
        "plot_path": ROOT / "wfi01_y_binned_candlesticks.png",
    },
}


def parse_mast_time(value):
    value = str(value).rstrip("Z")
    date_part, separator, fraction = value.partition(".")
    result = datetime.fromisoformat(date_part)
    if separator:
        result += timedelta(microseconds=int((fraction + "000000")[:6]))
    return result


def quantile_row(values):
    if values.size == 0:
        return np.full(5, np.nan)
    return np.percentile(values, [5, 25, 50, 75, 95])


def make_bins(length):
    return [
        (start, min(start + BIN_SIZE, length))
        for start in range(0, length, BIN_SIZE)
    ]


def take_axis_region(array, axis_name, start, stop):
    if AXES[axis_name]["dimension"] == 1:
        return array[:, start:stop]
    return array[start:stop, :]


def calculate_led(rows, led):
    rows = sorted(rows, key=lambda row: row["exposure_start_time"])
    first_path = DATA_DIR / f'{rows[0]["fileSetName"]}_cal.asdf'
    with asdf.open(first_path, lazy_load=True, ignore_unrecognized_tag=True) as af:
        first_data = np.asarray(af.tree["roman"]["data"])
        first_dq = np.asarray(af.tree["roman"]["dq"])

    first_valid = np.isfinite(first_data) & (first_dq == 0)
    baseline_valid = first_valid & (first_data > 0)
    first_median = float(np.median(first_data[first_valid]))
    bins_by_axis = {
        axis_name: make_bins(first_data.shape[details["dimension"]])
        for axis_name, details in AXES.items()
    }
    start_time = parse_mast_time(rows[0]["exposure_start_time"])
    result = {
        "led": led,
        "rows": rows,
        "bins": bins_by_axis,
        "records": {axis_name: [] for axis_name in AXES},
    }

    for row in rows:
        path = DATA_DIR / f'{row["fileSetName"]}_cal.asdf'
        with asdf.open(path, lazy_load=True, ignore_unrecognized_tag=True) as af:
            model = af.tree["roman"]
            data = np.asarray(model["data"])
            dq = np.asarray(model["dq"])

        frame_valid = np.isfinite(data) & (dq == 0)
        frame_median = float(np.median(data[frame_valid]))
        valid = baseline_valid & frame_valid
        hours = (
            parse_mast_time(row["exposure_start_time"]) - start_time
        ).total_seconds() / 3600

        for axis_name in AXES:
            raw_quantiles = []
            normalized_quantiles = []
            valid_counts = []
            for start, stop in bins_by_axis[axis_name]:
                data_region = take_axis_region(data, axis_name, start, stop)
                first_region = take_axis_region(first_data, axis_name, start, stop)
                valid_region = take_axis_region(valid, axis_name, start, stop)
                current_values = data_region[valid_region]
                first_values = first_region[valid_region]
                raw_change = (current_values / first_values - 1.0) * 100.0
                normalized_change = (
                    current_values / frame_median - first_values / first_median
                ) * 100.0
                raw_quantiles.append(quantile_row(raw_change))
                normalized_quantiles.append(quantile_row(normalized_change))
                valid_counts.append(int(valid_region.sum()))

            result["records"][axis_name].append(
                {
                    "hours": hours,
                    "raw_quantiles": np.asarray(raw_quantiles),
                    "normalized_quantiles": np.asarray(normalized_quantiles),
                    "valid_counts": np.asarray(valid_counts),
                    "frame_median": frame_median,
                }
            )

    return result


with INDEX_PATH.open(newline="") as handle:
    indexed_rows = list(csv.DictReader(handle))

results = []
for led in range(1, 7):
    rows = [row for row in indexed_rows if int(row["led"]) == led]
    result = calculate_led(rows, led)
    results.append(result)
    final_x = result["records"]["x"][-1]
    final_y = result["records"]["y"][-1]
    print(
        f"LED {led}: final x-bin median range="
        f"{np.nanmin(final_x['raw_quantiles'][:, 2]):.3f}% to "
        f"{np.nanmax(final_x['raw_quantiles'][:, 2]):.3f}%; "
        f"final y-bin median range="
        f"{np.nanmin(final_y['raw_quantiles'][:, 2]):.3f}% to "
        f"{np.nanmax(final_y['raw_quantiles'][:, 2]):.3f}%"
    )


fieldnames = [
    "led",
    "bin_index",
    "pixel_start",
    "pixel_end_exclusive",
    "pixel_center",
    "repeat",
    "time_hours_since_first",
    "valid_pixels",
    "raw_p05",
    "raw_p25",
    "raw_median",
    "raw_p75",
    "raw_p95",
    "normalized_p05",
    "normalized_p25",
    "normalized_median",
    "normalized_p75",
    "normalized_p95",
    "frame_median",
]

for axis_name, details in AXES.items():
    with details["csv_path"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for repeat, record in enumerate(result["records"][axis_name]):
                for bin_index, ((start, stop), raw, normalized, count) in enumerate(
                    zip(
                        result["bins"][axis_name],
                        record["raw_quantiles"],
                        record["normalized_quantiles"],
                        record["valid_counts"],
                    )
                ):
                    writer.writerow(
                        {
                            "led": result["led"],
                            "bin_index": bin_index,
                            "pixel_start": start,
                            "pixel_end_exclusive": stop,
                            "pixel_center": f"{(start + stop - 1) / 2:.1f}",
                            "repeat": repeat,
                            "time_hours_since_first": f"{record['hours']:.6f}",
                            "valid_pixels": int(count),
                            "raw_p05": f"{raw[0]:.6f}",
                            "raw_p25": f"{raw[1]:.6f}",
                            "raw_median": f"{raw[2]:.6f}",
                            "raw_p75": f"{raw[3]:.6f}",
                            "raw_p95": f"{raw[4]:.6f}",
                            "normalized_p05": f"{normalized[0]:.6f}",
                            "normalized_p25": f"{normalized[1]:.6f}",
                            "normalized_median": f"{normalized[2]:.6f}",
                            "normalized_p75": f"{normalized[3]:.6f}",
                            "normalized_p95": f"{normalized[4]:.6f}",
                            "frame_median": f"{record['frame_median']:.6f}",
                        }
                    )


colors = {led: plt.get_cmap("tab10")(led - 1) for led in range(1, 7)}


def plot_axis(axis_name, details):
    axis_label = details["label"]
    dimension = results[0]["bins"][axis_name][-1][1]
    final_key = "records"
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.subplots_adjust(top=0.82, bottom=0.17, wspace=0.12)
    fig.suptitle(
        f"WFI01 RCS-flat candidate: 100-pixel {axis_label} bins\n"
        "First-to-last pixel-change distributions; every candle uses pixels within one bin",
        fontsize=16,
    )

    for plot_axis, key, title in (
        (axes[0], "raw_quantiles", "Raw per-pixel change from first exposure"),
        (
            axes[1],
            "normalized_quantiles",
            "Change after frame-median normalization",
        ),
    ):
        all_values = np.concatenate(
            [result[final_key][axis_name][-1][key].ravel() for result in results]
        )
        all_values = all_values[np.isfinite(all_values)]
        lower = float(np.min(all_values))
        upper = float(np.max(all_values))
        padding = max((upper - lower) * 0.06, 0.1)
        plot_axis.set_ylim(lower - padding, upper + padding)

        for result in results:
            led = result["led"]
            record = result[final_key][axis_name][-1]
            offset = (led - 3.5) * 10.0
            for (start, stop), values in zip(
                result["bins"][axis_name], record[key]
            ):
                q05, q25, median, q75, q95 = values
                x = (start + stop - 1) / 2 + offset
                color = colors[led]
                plot_axis.vlines(x, q05, q95, color=color, linewidth=1.0)
                plot_axis.vlines(x, q25, q75, color=color, linewidth=7.0, alpha=0.9)
                plot_axis.plot(
                    x,
                    median,
                    marker="_",
                    markersize=9,
                    color="black",
                    markeredgewidth=1.2,
                )
            plot_axis.plot([], [], color=colors[led], linewidth=5.0, label=f"LED {led}")

        plot_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        plot_axis.set_title(title)
        plot_axis.set_xlabel(f"{axis_label} bin center")
        plot_axis.set_ylabel("Pixel change (%)")
        plot_axis.set_xlim(-60, dimension + 60)
        plot_axis.set_xticks(np.arange(0, dimension + 1, 500))
        plot_axis.grid(axis="x", color="0.85", linewidth=0.5)
        plot_axis.legend(frameon=False, ncol=3)

    fig.text(
        0.5,
        0.025,
        "Final repeat shown. Whisker=5th-95th percentile, body=25th-75th percentile, "
        "black tick=median. CSV contains all 13 repeats for every bin.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(details["plot_path"], dpi=180)
    plt.close(fig)
    print(f"Wrote {details['csv_path']}")
    print(f"Wrote {details['plot_path']}")


for axis_name, details in AXES.items():
    plot_axis(axis_name, details)
