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
SUMMARY_PATH = ROOT / "wfi01_all_led_summary.csv"
COMBINED_PLOT_PATH = ROOT / "wfi01_all_led_flat_timeseries.png"


def parse_mast_time(value):
    value = str(value).rstrip("Z")
    date_part, separator, fraction = value.partition(".")
    result = datetime.fromisoformat(date_part)
    if separator:
        result += timedelta(microseconds=int((fraction + "000000")[:6]))
    return result


def analyze_led(rows, led):
    rows = sorted(rows, key=lambda row: row["exposure_start_time"])
    start_time = parse_mast_time(rows[0]["exposure_start_time"])
    times_hours = np.asarray(
        [
            (parse_mast_time(row["exposure_start_time"]) - start_time).total_seconds()
            / 3600
            for row in rows
        ],
        dtype=float,
    )
    times_days = times_hours / 24.0

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

    normalized_change = np.where(stable, last_minus_first, np.nan)
    column_profile = np.nanmedian(normalized_change, axis=0)
    row_profile = np.nanmedian(normalized_change, axis=1)
    column_axis = np.linspace(0.0, 1.0, column_profile.size)
    row_axis = np.linspace(0.0, 1.0, row_profile.size)
    column_fit = np.polyfit(column_axis[np.isfinite(column_profile)], column_profile[np.isfinite(column_profile)], 1)
    row_fit = np.polyfit(row_axis[np.isfinite(row_profile)], row_profile[np.isfinite(row_profile)], 1)

    result = {
        "led": int(led),
        "rows": rows,
        "times_hours": times_hours,
        "data_medians": data_medians,
        "relative_medians": relative_medians,
        "valid_fractions": valid_fractions,
        "stable": stable,
        "last_minus_first": last_minus_first,
        "pixel_slope": pixel_slope,
        "stable_change": stable_change,
        "stable_slope": stable_slope,
        "change_p1": float(change_p1),
        "change_p99": float(change_p99),
        "slope_p1": float(slope_p1),
        "slope_p99": float(slope_p99),
        "column_gradient": float(column_fit[0]),
        "row_gradient": float(row_fit[0]),
    }

    led_summary_path = ROOT / f"wfi01_led{led}_flat_timeseries.csv"
    with led_summary_path.open("w", newline="") as handle:
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

    change_limit = max(abs(change_p1), abs(change_p99), 0.1)
    slope_limit = max(abs(slope_p1), abs(slope_p99), 0.1)
    median_fit = np.polyfit(times_hours / 24.0, relative_medians, 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(
        f"WFI01 RCS-flat candidate time series: LED {led}\n"
        f"{len(rows)} same-channel Level-2 exposures, normalized per frame",
        fontsize=15,
    )

    ax = axes[0, 0]
    ax.plot(times_hours, relative_medians, "o-", color="#2f6690")
    ax.plot(
        times_hours,
        np.polyval(median_fit, times_hours / 24.0),
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
    image = ax.imshow(
        np.ma.masked_where(~stable, last_minus_first),
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
    image = ax.imshow(
        np.ma.masked_where(~stable, pixel_slope),
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
    ax.hist(
        stable_change,
        bins=250,
        range=(-max(change_limit, 2), max(change_limit, 2)),
        color="#2f6690",
    )
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
        f"column gradient = {column_fit[0]:.4f}% across detector",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    led_plot_path = ROOT / f"wfi01_led{led}_flat_timeseries.png"
    fig.savefig(led_plot_path, dpi=180)
    plt.close(fig)
    return result


with INDEX_PATH.open(newline="") as handle:
    indexed_rows = list(csv.DictReader(handle))
led_results = []
for led in sorted({row["led"] for row in indexed_rows}, key=int):
    rows = [row for row in indexed_rows if row["led"] == led]
    result = analyze_led(rows, led)
    led_results.append(result)
    print(
        f"LED {led}: {len(rows)} frames, "
        f"raw net={result['relative_medians'][-1]:.6f}%, "
        f"raw slope={np.polyfit(result['times_hours'] / 24.0, result['relative_medians'], 1)[0]:.6f}%/day, "
        f"normalized median={np.median(result['stable_change']):.6f}%, "
        f"column gradient={result['column_gradient']:.6f}%"
    )

with SUMMARY_PATH.open("w", newline="") as handle:
    fieldnames = [
        "led",
        "n_frames",
        "first_median",
        "last_median",
        "raw_net_percent",
        "raw_slope_percent_per_day",
        "normalized_change_median_percent",
        "normalized_change_p1_percent",
        "normalized_change_p99_percent",
        "normalized_slope_median_percent_per_day",
        "column_gradient_percent_across_detector",
        "row_gradient_percent_across_detector",
        "stable_pixels",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for result in led_results:
        raw_fit = np.polyfit(result["times_hours"] / 24.0, result["relative_medians"], 1)
        writer.writerow(
            {
                "led": result["led"],
                "n_frames": len(result["rows"]),
                "first_median": f'{result["data_medians"][0]:.6f}',
                "last_median": f'{result["data_medians"][-1]:.6f}',
                "raw_net_percent": f'{result["relative_medians"][-1]:.6f}',
                "raw_slope_percent_per_day": f"{raw_fit[0]:.6f}",
                "normalized_change_median_percent": f'{np.median(result["stable_change"]):.6f}',
                "normalized_change_p1_percent": f'{result["change_p1"]:.6f}',
                "normalized_change_p99_percent": f'{result["change_p99"]:.6f}',
                "normalized_slope_median_percent_per_day": f'{np.median(result["stable_slope"]):.6f}',
                "column_gradient_percent_across_detector": f'{result["column_gradient"]:.6f}',
                "row_gradient_percent_across_detector": f'{result["row_gradient"]:.6f}',
                "stable_pixels": int(result["stable"].sum()),
            }
        )

fig, axes = plt.subplots(3, 3, figsize=(16, 16), constrained_layout=True)
flat_axes = axes.ravel()
fig.suptitle(
    "WFI01 RCS-flat candidate: all six LED channels\n"
    "Detector-wide trends and frame-normalized last-minus-first maps",
    fontsize=16,
)

ax = flat_axes[0]
for result in led_results:
    ax.plot(
        result["times_hours"],
        result["relative_medians"],
        "o-",
        markersize=3,
        label=f'LED {result["led"]}',
    )
ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
ax.set_title("Detector-wide signal")
ax.set_xlabel("Hours since first channel exposure")
ax.set_ylabel("Median change from first (%)")
ax.legend(frameon=False, ncol=2)

all_changes = np.concatenate([result["stable_change"] for result in led_results])
common_limit = max(abs(np.percentile(all_changes, 1)), abs(np.percentile(all_changes, 99)), 0.1)
for axis, result in zip(flat_axes[1:], led_results):
    stable_change = result["last_minus_first"]
    stable = result["stable"]
    image = axis.imshow(
        np.ma.masked_where(~stable, stable_change),
        origin="lower",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-common_limit, vcenter=0.0, vmax=common_limit),
        interpolation="nearest",
        rasterized=True,
    )
    axis.set_title(
        f'LED {result["led"]}: '
        f'{result["relative_medians"][-1]:.2f}% raw, '
        f'{result["column_gradient"]:.2f}% column gradient'
    )
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")

for axis in flat_axes[1 + len(led_results) :]:
    axis.axis("off")

fig.colorbar(image, ax=axes[1:, :], fraction=0.02, pad=0.02, label="Last - first (%)")
fig.savefig(COMBINED_PLOT_PATH, dpi=180)
plt.close(fig)

print(f"Wrote {SUMMARY_PATH}")
print(f"Wrote {COMBINED_PLOT_PATH}")
