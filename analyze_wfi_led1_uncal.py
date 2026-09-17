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
SUMMARY_PATH = ROOT / "wfi01_led1_uncal_ramp_summary.csv"
PLOT_PATH = ROOT / "wfi01_led1_uncal_ramp_check.png"
READ_POSITIONS = np.asarray([1, 4, 7, 10, 13], dtype=float)


def parse_mast_time(value):
    value = str(value).rstrip("Z")
    date_part, separator, fraction = value.partition(".")
    result = datetime.fromisoformat(date_part)
    if separator:
        result += timedelta(microseconds=int((fraction + "000000")[:6]))
    return result


with INDEX_PATH.open(newline="") as handle:
    rows = sorted(
        [row for row in csv.DictReader(handle) if row["led"] == "1"],
        key=lambda row: row["exposure_start_time"],
    )

start_time = parse_mast_time(rows[0]["exposure_start_time"])
times_hours = np.asarray(
    [
        (parse_mast_time(row["exposure_start_time"]) - start_time).total_seconds() / 3600
        for row in rows
    ]
)

read_medians = []
ramp_slopes = []
for row in rows:
    path = DATA_DIR / f'{row["fileSetName"]}_uncal.asdf'
    with asdf.open(path, lazy_load=True, ignore_unrecognized_tag=True) as af:
        data = np.asarray(af.tree["roman"]["data"])
    medians = np.median(data.reshape(data.shape[0], -1), axis=1)
    slope = np.polyfit(READ_POSITIONS, medians, 1)[0]
    read_medians.append(medians)
    ramp_slopes.append(slope)

read_medians = np.asarray(read_medians)
ramp_slopes = np.asarray(ramp_slopes)
relative_slopes = (ramp_slopes / ramp_slopes[0] - 1.0) * 100.0

with SUMMARY_PATH.open("w", newline="") as handle:
    fieldnames = [
        "fileSetName",
        "exposure_start_time",
        "time_hours_since_first",
        "read1_median",
        "read4_median",
        "read7_median",
        "read10_median",
        "read13_median",
        "ramp_slope_dn_per_read",
        "ramp_slope_relative_percent",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row, hours, medians, slope, relative in zip(
        rows, times_hours, read_medians, ramp_slopes, relative_slopes
    ):
        writer.writerow(
            {
                "fileSetName": row["fileSetName"],
                "exposure_start_time": row["exposure_start_time"],
                "time_hours_since_first": f"{hours:.6f}",
                "read1_median": f"{medians[0]:.6f}",
                "read4_median": f"{medians[1]:.6f}",
                "read7_median": f"{medians[2]:.6f}",
                "read10_median": f"{medians[3]:.6f}",
                "read13_median": f"{medians[4]:.6f}",
                "ramp_slope_dn_per_read": f"{slope:.6f}",
                "ramp_slope_relative_percent": f"{relative:.6f}",
            }
        )

calibrated_path = ROOT / "wfi01_led1_flat_timeseries.csv"
with calibrated_path.open(newline="") as handle:
    calibrated = list(csv.DictReader(handle))
calibrated_relative = np.asarray([float(row["relative_median_percent"]) for row in calibrated])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
fig.suptitle(
    "WFI01 LED 1 raw-ramp validation\n"
    "The uncalibrated ramp-slope proxy is compared with the Level-2 calibrated signal",
    fontsize=14,
)

ax = axes[0]
for index, position in enumerate(READ_POSITIONS):
    ax.plot(times_hours, read_medians[:, index], "o-", label=f"Read {int(position)}")
ax.set_title("Median raw ramp values")
ax.set_xlabel("Hours since first exposure")
ax.set_ylabel("Raw detector units")
ax.legend(frameon=False, ncol=2)

ax = axes[1]
ax.plot(times_hours, relative_slopes, "o-", color="#2f6690", label="Uncalibrated ramp slope")
ax.plot(times_hours, calibrated_relative, "s--", color="#c1121f", label="Calibrated data median")
ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
ax.set_title("Relative change from first exposure")
ax.set_xlabel("Hours since first exposure")
ax.set_ylabel("Change (%)")
ax.legend(frameon=False)
ax.text(
    0.03,
    0.97,
    f"uncal ramp net = {relative_slopes[-1]:.3f}%\n"
    f"calibrated net = {calibrated_relative[-1]:.3f}%",
    transform=ax.transAxes,
    va="top",
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)

fig.savefig(PLOT_PATH, dpi=180)
plt.close(fig)

print(f"Uncalibrated ramp net change: {relative_slopes[-1]:.6f}%")
print(f"Calibrated median net change: {calibrated_relative[-1]:.6f}%")
print(f"Wrote {SUMMARY_PATH}")
print(f"Wrote {PLOT_PATH}")
