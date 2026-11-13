import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astroquery.mast import MastMissions


ROOT = Path(__file__).resolve().parent
TOKEN_PATH = Path("/Users/paterson.35/Downloads/mast_api_token.txt")
METADATA_PATH = ROOT / "all_wfi_rcs_flat_metadata.csv"
SUMMARY_PATH = ROOT / "all_wfi_rcs_flat_detector_summary.csv"
PLOT_PATH = ROOT / "all_wfi_rcs_flat_detector_trends.png"

token = TOKEN_PATH.read_text().strip()
missions = MastMissions(mission="roman")
missions.login(token=token)

select_cols = [
    "fileSetName",
    "detector",
    "led",
    "bank",
    "program",
    "exposure_type",
    "optical_element",
    "exposure_start_time",
    "exposure_end_time",
    "image_median",
    "image_rms",
    "good_pixel_fraction",
    "visit_id",
]
metadata = missions.query_criteria(
    exposure_type="WFI_FLAT",
    optical_element="DARK",
    product_type="l2",
    program="1039",
    select_cols=select_cols,
    sort_by="exposure_start_time",
    sort_desc=False,
    limit=5000,
)

with METADATA_PATH.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=select_cols)
    writer.writeheader()
    for row in metadata:
        writer.writerow({name: str(row[name]) for name in select_cols})

rows = []
with METADATA_PATH.open(newline="") as handle:
    for row in csv.DictReader(handle):
        row["image_median"] = float(row["image_median"])
        rows.append(row)

groups = {}
for row in rows:
    groups.setdefault((row["detector"], int(row["led"])), []).append(row)

results = []
for (detector, led), detector_rows in sorted(groups.items()):
    detector_rows.sort(key=lambda row: row["exposure_start_time"])
    medians = np.asarray([row["image_median"] for row in detector_rows])
    relative = (medians / medians[0] - 1.0) * 100.0
    times = np.arange(len(relative), dtype=float)
    slope = np.polyfit(times / 24.0, relative, 1)[0]
    results.append(
        {
            "detector": detector,
            "led": led,
            "n_frames": len(detector_rows),
            "first_median": medians[0],
            "last_median": medians[-1],
            "raw_net_percent": relative[-1],
            "raw_slope_percent_per_day": slope,
        }
    )

with SUMMARY_PATH.open("w", newline="") as handle:
    fieldnames = [
        "detector",
        "led",
        "n_frames",
        "first_median",
        "last_median",
        "raw_net_percent",
        "raw_slope_percent_per_day",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "detector": result["detector"],
                "led": result["led"],
                "n_frames": result["n_frames"],
                "first_median": f'{result["first_median"]:.6f}',
                "last_median": f'{result["last_median"]:.6f}',
                "raw_net_percent": f'{result["raw_net_percent"]:.6f}',
                "raw_slope_percent_per_day": f'{result["raw_slope_percent_per_day"]:.6f}',
            }
        )

fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True, constrained_layout=True)
fig.suptitle(
    "WFI RCS-flat candidate archive medians across all 18 detectors\n"
    "Each faint line is one detector; thick line is the detector median",
    fontsize=16,
)

for axis, led in zip(axes.ravel(), range(1, 7)):
    led_results = [result for result in results if result["led"] == led]
    detector_series = []
    for result in led_results:
        detector_rows = groups[(result["detector"], led)]
        detector_rows.sort(key=lambda row: row["exposure_start_time"])
        medians = np.asarray([row["image_median"] for row in detector_rows])
        detector_series.append((medians / medians[0] - 1.0) * 100.0)

    series = np.asarray(detector_series)
    hours = np.arange(series.shape[1])
    axis.plot(hours, series.T, color="#9aa0a6", alpha=0.5, linewidth=0.8)
    axis.plot(hours, np.median(series, axis=0), color="#c1121f", linewidth=2.5, label="18-detector median")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    net = series[:, -1]
    axis.set_title(f"LED {led}: median net {np.median(net):.2f}%\nrange {net.min():.2f}% to {net.max():.2f}%")
    axis.set_xlabel("Sequence index")
    axis.set_ylabel("Median change from first (%)")
    axis.legend(frameon=False, fontsize=8)

fig.savefig(PLOT_PATH, dpi=180)
plt.close(fig)

for led in range(1, 7):
    net = np.asarray([result["raw_net_percent"] for result in results if result["led"] == led])
    print(
        f"LED {led}: detector net median={np.median(net):.6f}%, "
        f"std={np.std(net):.6f}%, min={net.min():.6f}%, max={net.max():.6f}%"
    )
print(f"Wrote {METADATA_PATH}")
print(f"Wrote {SUMMARY_PATH}")
print(f"Wrote {PLOT_PATH}")
