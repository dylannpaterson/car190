import csv
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import asdf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "wfi01_rcs_flat_index.csv"
DATA_DIR = ROOT / "MAST/Roman"
SUMMARY_PATH = ROOT / "wfi01_led1_flat_timeseries.csv"
PLOT_PATH = ROOT / "wfi01_led1_flat_timeseries.png"


def parse_mast_time(value):
    value = str(value).rstrip("Z")
    date_part, separator, fraction = value.partition(".")
    result = datetime.fromisoformat(date_part)
    if separator:
        result += timedelta(microseconds=int((fraction + "000000")[:6]))
    return result


with INDEX_PATH.open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["led"] == "1"]
rows.sort(key=lambda row: row["exposure_start_time"])

if not rows:
    raise RuntimeError("No LED 1 flat frames found in the MAST index")

start_time = parse_mast_time(rows[0]["exposure_start_time"])
times_hours = [
    (parse_mast_time(row["exposure_start_time"]) - start_time).total_seconds() / 3600
    for row in rows
]
times_days = np.asarray(times_hours, dtype=float) / 24.0

data_medians = []
valid_fractions = []
first_normalized = None
last_normalized = None
sum_n = None
sum_x = None
sum_y = None
sum_xx = None
sum_xy = None

for index, row in enumerate(rows):
    path = DATA_DIR / f'{row["fileSetName"]}_cal.asdf'
    if not path.exists():
        raise FileNotFoundError(path)

    with asdf.open(path, lazy_load=True, ignore_unrecognized_tag=True) as af:
        model = af.tree["roman"]
        data = np.asarray(model["data"])
        dq = np.asarray(model["dq"])

    valid = np.isfinite(data) & (dq == 0)
    median = float(np.median(data[valid]))
    normalized = data / median
    data_medians.append(median)
    valid_fractions.append(float(valid.mean()))

    if first_normalized is None:
        first_normalized = np.full(data.shape, np.nan, dtype=np.float32)
        sum_n = np.zeros(data.shape, dtype=np.int16)
        sum_x = np.zeros(data.shape, dtype=np.float64)
        sum_y = np.zeros(data.shape, dtype=np.float64)
        sum_xx = np.zeros(data.shape, dtype=np.float64)
        sum_xy = np.zeros(data.shape, dtype=np.float64)

    if index == 0:
        first_normalized[valid] = normalized[valid]
    last_normalized = np.full(data.shape, np.nan, dtype=np.float32)
    last_normalized[valid] = normalized[valid]

    x = times_days[index]
    sum_n += valid
    sum_x += np.where(valid, x, 0.0)
    sum_y += np.where(valid, normalized, 0.0)
    sum_xx += np.where(valid, x * x, 0.0)
    sum_xy += np.where(valid, normalized * x, 0.0)

data_medians = np.asarray(data_medians)
valid_fractions = np.asarray(valid_fractions)
relative_medians = (data_medians / data_medians[0] - 1.0) * 100.0

stable = (sum_n == len(rows)) & np.isfinite(first_normalized) & np.isfinite(last_normalized)
last_minus_first = (last_normalized - first_normalized) * 100.0
denominator = sum_n * sum_xx - sum_x * sum_x
pixel_slope = np.full(first_normalized.shape, np.nan, dtype=np.float64)
pixel_slope = np.divide(
    sum_n * sum_xy - sum_x * sum_y,
    denominator,
    out=pixel_slope,
    where=(stable & (denominator > 0)),
)
pixel_slope *= 100.0

stable_change = last_minus_first[stable]
stable_slope = pixel_slope[stable]
change_p1, change_p99 = np.percentile(stable_change, [1, 99])
slope_p1, slope_p99 = np.percentile(stable_slope, [1, 99])
change_limit = max(abs(change_p1), abs(change_p99), 0.1)
slope_limit = max(abs(slope_p1), abs(slope_p99), 0.1)

median_fit = np.polyfit(times_days, relative_medians, 1)

with SUMMARY_PATH.open("w", newline="") as handle:
    fieldnames = [
        "fileSetName",
        "exposure_start_time",
        "exposure_end_time",
        "led",
        "bank",
        "time_hours_since_first",
        "data_median",
        "relative_median_percent",
        "valid_fraction",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row, hours, median, relative, fraction in zip(
        rows, times_hours, data_medians, relative_medians, valid_fractions
    ):
        writer.writerow(
            {
                "fileSetName": row["fileSetName"],
                "exposure_start_time": row["exposure_start_time"],
                "exposure_end_time": row["exposure_end_time"],
                "led": row["led"],
                "bank": row["bank"],
                "time_hours_since_first": f"{hours:.6f}",
                "data_median": f"{median:.6f}",
                "relative_median_percent": f"{relative:.6f}",
                "valid_fraction": f"{fraction:.9f}",
            }
        )

fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
fig.suptitle(
    "WFI01 RCS-flat candidate time series: LED 1\n"
    "13 same-channel Level-2 exposures, normalized per frame",
    fontsize=15,
)

ax = axes[0, 0]
ax.plot(times_hours, relative_medians, "o-", color="#2f6690")
ax.plot(
    times_hours,
    np.polyval(median_fit, times_days),
    color="#c1121f",
    linestyle="--",
    label=f"linear trend = {median_fit[0]:.3f}%/day",
)
ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
ax.set_title("Detector-wide signal")
ax.set_xlabel("Hours since first exposure")
ax.set_ylabel("Median change from first (%)")
ax.legend(frameon=False)
ax.text(
    0.03,
    0.97,
    f"first median = {data_medians[0]:.3f}\n"
    f"last median = {data_medians[-1]:.3f}\n"
    f"net change = {relative_medians[-1]:.3f}%",
    transform=ax.transAxes,
    va="top",
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)

ax = axes[0, 1]
change_image = np.ma.masked_where(~stable, last_minus_first)
image = ax.imshow(
    change_image,
    origin="lower",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-change_limit, vcenter=0.0, vmax=change_limit),
    interpolation="nearest",
    rasterized=True,
)
ax.set_title("Pixel change after frame-median normalization")
ax.set_xlabel("Detector column")
ax.set_ylabel("Detector row")
cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, extend="both")
cbar.set_label("Last - first (%)")

ax = axes[1, 0]
slope_image = np.ma.masked_where(~stable, pixel_slope)
image = ax.imshow(
    slope_image,
    origin="lower",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-slope_limit, vcenter=0.0, vmax=slope_limit),
    interpolation="nearest",
    rasterized=True,
)
ax.set_title("Per-pixel normalized trend")
ax.set_xlabel("Detector column")
ax.set_ylabel("Detector row")
cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, extend="both")
cbar.set_label("Percent per day")

ax = axes[1, 1]
ax.hist(stable_change, bins=250, range=(-max(change_limit, 2), max(change_limit, 2)), color="#2f6690")
ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
ax.set_title("Distribution of normalized pixel changes")
ax.set_xlabel("Last - first (%)")
ax.set_ylabel("Pixels")
ax.text(
    0.03,
    0.97,
    f"stable pixels = {int(stable.sum()):,}\n"
    f"median = {np.median(stable_change):.4f}%\n"
    f"1st-99th = {change_p1:.4f}% to {change_p99:.4f}%\n"
    f"slope median = {np.median(stable_slope):.4f}%/day",
    transform=ax.transAxes,
    va="top",
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)

fig.savefig(PLOT_PATH, dpi=180)
plt.close(fig)

print(f"Processed {len(rows)} frames")
print(f"Detector-wide net change: {relative_medians[-1]:.6f}%")
print(f"Detector-wide linear trend: {median_fit[0]:.6f}%/day")
print(f"Stable-pixel median normalized change: {np.median(stable_change):.6f}%")
print(f"Stable-pixel normalized change 1st-99th: {change_p1:.6f}% to {change_p99:.6f}%")
print(f"Wrote {SUMMARY_PATH}")
print(f"Wrote {PLOT_PATH}")
