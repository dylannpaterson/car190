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
QUANTILE_PATH = ROOT / "wfi01_all_pixel_candlesticks.csv"
PLOT_PATH = ROOT / "wfi01_all_pixel_candlesticks.png"


def parse_mast_time(value):
    value = str(value).rstrip("Z")
    date_part, separator, fraction = value.partition(".")
    result = datetime.fromisoformat(date_part)
    if separator:
        result += timedelta(microseconds=int((fraction + "000000")[:6]))
    return result


def quantile_row(values):
    return np.percentile(values, [5, 25, 50, 75, 95])


with INDEX_PATH.open(newline="") as handle:
    indexed_rows = list(csv.DictReader(handle))

all_results = []
for led in range(1, 7):
    rows = sorted(
        [row for row in indexed_rows if int(row["led"]) == led],
        key=lambda row: row["exposure_start_time"],
    )

    first_path = DATA_DIR / f'{rows[0]["fileSetName"]}_cal.asdf'
    with asdf.open(first_path, lazy_load=True, ignore_unrecognized_tag=True) as af:
        first_data = np.asarray(af.tree["roman"]["data"])
        first_dq = np.asarray(af.tree["roman"]["dq"])
    first_valid = np.isfinite(first_data) & (first_dq == 0)
    baseline_valid = first_valid & (first_data > 0)
    first_median = float(np.median(first_data[first_valid]))

    result = {
        "led": led,
        "rows": rows,
        "raw_quantiles": [],
        "normalized_quantiles": [],
        "valid_counts": [],
        "frame_medians": [],
        "hours": [],
    }
    start_time = parse_mast_time(rows[0]["exposure_start_time"])

    for row in rows:
        path = DATA_DIR / f'{row["fileSetName"]}_cal.asdf'
        with asdf.open(path, lazy_load=True, ignore_unrecognized_tag=True) as af:
            model = af.tree["roman"]
            data = np.asarray(model["data"])
            dq = np.asarray(model["dq"])

        frame_valid = np.isfinite(data) & (dq == 0)
        frame_median = float(np.median(data[frame_valid]))
        valid = baseline_valid & frame_valid

        # Relative ratios are well-defined after excluding the few zero-baseline pixels.
        raw_change = (data[valid] / first_data[valid] - 1.0) * 100.0
        normalized_change = (
            data[valid] / frame_median - first_data[valid] / first_median
        ) * 100.0
        result["raw_quantiles"].append(quantile_row(raw_change))
        result["normalized_quantiles"].append(quantile_row(normalized_change))
        result["valid_counts"].append(int(valid.sum()))
        result["frame_medians"].append(frame_median)
        result["hours"].append(
            (parse_mast_time(row["exposure_start_time"]) - start_time).total_seconds() / 3600
        )

    result["raw_quantiles"] = np.asarray(result["raw_quantiles"])
    result["normalized_quantiles"] = np.asarray(result["normalized_quantiles"])
    result["valid_counts"] = np.asarray(result["valid_counts"])
    result["frame_medians"] = np.asarray(result["frame_medians"])
    result["hours"] = np.asarray(result["hours"])
    all_results.append(result)
    print(
        f"LED {led}: all-pixel raw net median="
        f"{result['raw_quantiles'][-1, 2]:.6f}%, "
        f"valid pixels={result['valid_counts'][-1]:,}"
    )

with QUANTILE_PATH.open("w", newline="") as handle:
    fieldnames = [
        "led",
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
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for result in all_results:
        for repeat, (hours, raw, normalized, count, frame_median) in enumerate(
            zip(
                result["hours"],
                result["raw_quantiles"],
                result["normalized_quantiles"],
                result["valid_counts"],
                result["frame_medians"],
            )
        ):
            writer.writerow(
                {
                    "led": result["led"],
                    "repeat": repeat,
                    "time_hours_since_first": f"{hours:.6f}",
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
                    "frame_median": f"{frame_median:.6f}",
                }
            )

colors = {led: plt.get_cmap("tab10")(led - 1) for led in range(1, 7)}
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.subplots_adjust(top=0.82, bottom=0.17, wspace=0.12)
fig.suptitle(
    "WFI01 RCS-flat candidate: all-pixel candlesticks by LED\n"
    "Every candle uses all DQ-clean pixels with a positive first-frame baseline",
    fontsize=16,
)


def draw_candlesticks(axis, key, title):
    for result in all_results:
        led = result["led"]
        quantiles = result[key]
        offset = (led - 3.5) * 0.11
        for repeat, values in enumerate(quantiles):
            q05, q25, median, q75, q95 = values
            x = repeat + offset
            color = colors[led]
            axis.vlines(x, q05, q95, color=color, linewidth=1.0)
            axis.vlines(x, q25, q75, color=color, linewidth=7.0, alpha=0.9)
            axis.plot(x, median, marker="_", markersize=9, color="black", markeredgewidth=1.2)
        axis.plot([], [], color=colors[led], linewidth=5.0, label=f"LED {led}")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Repeat number")
    axis.set_ylabel("Pixel change (%)")
    axis.set_xticks(range(13))
    axis.legend(frameon=False, ncol=3)


draw_candlesticks(
    axes[0],
    "raw_quantiles",
    "Raw per-pixel change from first exposure",
)
draw_candlesticks(
    axes[1],
    "normalized_quantiles",
    "Change after frame-median normalization",
)
fig.text(
    0.5,
    0.025,
    "Whisker=5th-95th percentile, body=25th-75th percentile, black tick=median. "
    "Quantiles are exact over approximately 15.8 million usable pixels per frame.",
    ha="center",
    fontsize=10,
)
fig.savefig(PLOT_PATH, dpi=180)
plt.close(fig)

print(f"Wrote {QUANTILE_PATH}")
print(f"Wrote {PLOT_PATH}")
