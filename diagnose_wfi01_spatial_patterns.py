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
SUMMARY_PATH = ROOT / "wfi01_spatial_pattern_summary.csv"
CORRELATION_PATH = ROOT / "wfi01_spatial_pattern_correlations.csv"
MAP_PLOT_PATH = ROOT / "wfi01_spatial_change_maps.png"
CORRELATION_PLOT_PATH = ROOT / "wfi01_spatial_change_correlations.png"
BLOCK_SIZE = 64


def parse_mast_time(value):
    value = str(value).rstrip("Z")
    date_part, separator, fraction = value.partition(".")
    result = datetime.fromisoformat(date_part)
    if separator:
        result += timedelta(microseconds=int((fraction + "000000")[:6]))
    return result


def load_image(path):
    with asdf.open(path, lazy_load=True, ignore_unrecognized_tag=True) as af:
        model = af.tree["roman"]
        data = np.asarray(model["data"])
        dq = np.asarray(model["dq"])
    valid = np.isfinite(data) & (dq == 0)
    median = float(np.median(data[valid]))
    return data, valid, median


def block_mean(values, valid):
    rows = values.shape[0] // BLOCK_SIZE
    columns = values.shape[1] // BLOCK_SIZE
    cropped = np.where(valid[: rows * BLOCK_SIZE, : columns * BLOCK_SIZE], values[: rows * BLOCK_SIZE, : columns * BLOCK_SIZE], np.nan)
    blocks = cropped.reshape(rows, BLOCK_SIZE, columns, BLOCK_SIZE)
    count = np.isfinite(blocks).sum(axis=(1, 3))
    total = np.nansum(blocks, axis=(1, 3))
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


with INDEX_PATH.open(newline="") as handle:
    indexed_rows = list(csv.DictReader(handle))

results = []
for led in range(1, 7):
    rows = sorted(
        [row for row in indexed_rows if int(row["led"]) == led],
        key=lambda row: row["exposure_start_time"],
    )
    first_data, first_valid, first_median = load_image(
        DATA_DIR / f'{rows[0]["fileSetName"]}_cal.asdf'
    )
    last_data, last_valid, last_median = load_image(
        DATA_DIR / f'{rows[-1]["fileSetName"]}_cal.asdf'
    )
    stable = first_valid & last_valid
    # Use additive fractional changes to avoid unstable ratios in very low-valued pixels.
    raw_change = ((last_data - first_data) / first_median) * 100.0
    normalized_change = (
        last_data / last_median - first_data / first_median
    ) * 100.0
    raw_map = block_mean(raw_change, stable)
    normalized_map = block_mean(normalized_change, stable)
    column_profile = np.nanmedian(normalized_map, axis=0)
    row_profile = np.nanmedian(normalized_map, axis=1)
    column_axis = np.linspace(0.0, 1.0, column_profile.size)
    row_axis = np.linspace(0.0, 1.0, row_profile.size)
    column_fit = np.polyfit(column_axis, column_profile, 1)
    row_fit = np.polyfit(row_axis, row_profile, 1)
    results.append(
        {
            "led": led,
            "first_median": first_median,
            "last_median": last_median,
        "raw_net_percent": (last_median / first_median - 1.0) * 100.0,
            "normalized_map": normalized_map,
            "raw_map": raw_map,
            "column_profile": column_profile,
            "row_profile": row_profile,
            "column_gradient": column_fit[0],
            "row_gradient": row_fit[0],
            "stable_pixels": int(stable.sum()),
            "start_time": rows[0]["exposure_start_time"],
            "end_time": rows[-1]["exposure_end_time"],
        }
    )

with SUMMARY_PATH.open("w", newline="") as handle:
    fieldnames = [
        "led",
        "start_time",
        "end_time",
        "first_median",
        "last_median",
        "raw_net_percent",
        "normalized_map_median_percent",
        "normalized_map_std_percent",
        "column_gradient_percent_across_detector",
        "row_gradient_percent_across_detector",
        "stable_pixels",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        map_values = result["normalized_map"][np.isfinite(result["normalized_map"])]
        writer.writerow(
            {
                "led": result["led"],
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "first_median": f'{result["first_median"]:.6f}',
                "last_median": f'{result["last_median"]:.6f}',
                "raw_net_percent": f'{result["raw_net_percent"]:.6f}',
                "normalized_map_median_percent": f"{np.median(map_values):.6f}",
                "normalized_map_std_percent": f"{np.std(map_values):.6f}",
                "column_gradient_percent_across_detector": f'{result["column_gradient"]:.6f}',
                "row_gradient_percent_across_detector": f'{result["row_gradient"]:.6f}',
                "stable_pixels": result["stable_pixels"],
            }
        )

correlations = np.eye(len(results))
for first in range(len(results)):
    for second in range(first):
        first_map = results[first]["normalized_map"]
        second_map = results[second]["normalized_map"]
        valid = np.isfinite(first_map) & np.isfinite(second_map)
        correlations[first, second] = correlations[second, first] = np.corrcoef(
            first_map[valid], second_map[valid]
        )[0, 1]

with CORRELATION_PATH.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["led"] + [f"led{led}" for led in range(1, 7)])
    for led, row in enumerate(correlations, start=1):
        writer.writerow([f"led{led}"] + [f"{value:.6f}" for value in row])

raw_values = np.concatenate(
    [result["raw_map"][np.isfinite(result["raw_map"])] for result in results]
)
normalized_values = np.concatenate(
    [result["normalized_map"][np.isfinite(result["normalized_map"])] for result in results]
)
raw_limit = max(abs(np.percentile(raw_values, 1)), abs(np.percentile(raw_values, 99)), 0.1)
normalized_limit = max(
    abs(np.percentile(normalized_values, 1)),
    abs(np.percentile(normalized_values, 99)),
    0.1,
)

fig, axes = plt.subplots(2, 6, figsize=(20, 7), constrained_layout=True)
fig.suptitle(
    "WFI01 first-to-last spatial changes, block-averaged over 64x64 pixels\n"
    "Top: raw change; bottom: change after removing each frame's detector median",
    fontsize=15,
)
for index, result in enumerate(results):
    for axis, values, limit, title in (
        (axes[0, index], result["raw_map"], raw_limit, "raw"),
        (axes[1, index], result["normalized_map"], normalized_limit, "frame-normalized"),
    ):
        image = axis.imshow(
            values,
            origin="lower",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            interpolation="nearest",
        )
        axis.set_title(f'LED {result["led"]}: {title}')
        axis.set_xlabel("64-pixel block column")
        if index == 0:
            axis.set_ylabel("64-pixel block row")
fig.colorbar(image, ax=axes, fraction=0.015, pad=0.01, label="Percent")
fig.savefig(MAP_PLOT_PATH, dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
fig.suptitle("WFI01 spatial-change tests for a wavelength-dependent throughput effect", fontsize=15)

image = axes[0].imshow(
    correlations,
    origin="lower",
    cmap="RdBu_r",
    vmin=-1.0,
    vmax=1.0,
    interpolation="nearest",
)
axes[0].set_title("Correlation of normalized spatial maps")
axes[0].set_xlabel("LED")
axes[0].set_ylabel("LED")
axes[0].set_xticks(range(6), range(1, 7))
axes[0].set_yticks(range(6), range(1, 7))
for row in range(6):
    for column in range(6):
        axes[0].text(column, row, f"{correlations[row, column]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04, label="Pearson r")

axis_values = np.linspace(0.0, 1.0, results[0]["column_profile"].size)
for result in results:
    axes[1].plot(axis_values, result["column_profile"], color=plt.get_cmap("tab10")(result["led"] - 1), label=f'LED {result["led"]}')
axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
axes[1].set_title("Frame-normalized column profiles")
axes[1].set_xlabel("Fractional detector column")
axes[1].set_ylabel("Last - first (%)")
axes[1].legend(frameon=False, ncol=2)

for result in results:
    axes[2].plot(axis_values, np.nanmean(result["raw_map"], axis=0), color=plt.get_cmap("tab10")(result["led"] - 1), label=f'LED {result["led"]}')
axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
axes[2].set_title("Raw column profiles")
axes[2].set_xlabel("Fractional detector column")
axes[2].set_ylabel("Last - first (%)")
axes[2].legend(frameon=False, ncol=2)

fig.savefig(CORRELATION_PLOT_PATH, dpi=180)
plt.close(fig)

for result in results:
    print(
        f'LED {result["led"]}: raw net={result["raw_net_percent"]:.6f}%, '
        f'normalized map std={np.nanstd(result["normalized_map"]):.6f}%, '
        f'column gradient={result["column_gradient"]:.6f}%, '
        f'row gradient={result["row_gradient"]:.6f}%'
    )
print(f"Wrote {SUMMARY_PATH}")
print(f"Wrote {CORRELATION_PATH}")
print(f"Wrote {MAP_PLOT_PATH}")
print(f"Wrote {CORRELATION_PLOT_PATH}")
