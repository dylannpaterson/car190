from pathlib import Path

import asdf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "MAST/Roman/r0103902001001002001_0001_wfi01_dark_cal.asdf"
OUTPUT_PATH = ROOT / "wfi01_dark_cal_flat_candidate.png"


with asdf.open(INPUT_PATH, lazy_load=True, ignore_unrecognized_tag=True) as af:
    model = af.tree["roman"]
    data = np.asarray(model["data"])
    dq = np.asarray(model["dq"])
    meta = model["meta"]

valid = np.isfinite(data) & (dq == 0)
values = data[valid]
median = float(np.median(values))
normalized = data / median
normalized_values = normalized[valid]
deviation = (normalized - 1.0) * 100.0
deviation_values = deviation[valid]
p1, p99 = np.percentile(values, [1, 99])
d1, d99 = np.percentile(deviation_values, [1, 99])
color_limit = max(abs(d1), abs(d99))

raw_image = np.ma.masked_where(~valid, data)
normalized_image = np.ma.masked_where(~valid, deviation)
dq_count = int(np.count_nonzero(dq))

row_medians = np.nanmedian(np.where(valid, data, np.nan), axis=1) / median
column_medians = np.nanmedian(np.where(valid, data, np.nan), axis=0) / median

fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
fig.suptitle(
    "WFI01 dark-named exposure: flat-field candidate\n"
    "r0103902001001002001_0001_wfi01_dark_cal.asdf",
    fontsize=15,
)

ax = axes[0, 0]
image = ax.imshow(
    raw_image,
    origin="lower",
    cmap="viridis",
    vmin=p1,
    vmax=p99,
    interpolation="nearest",
    rasterized=True,
)
ax.set_title(f"Calibrated data (1st-99th percentile: {p1:.1f}-{p99:.1f})")
ax.set_xlabel("Detector column")
ax.set_ylabel("Detector row")
cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Data value")

ax = axes[0, 1]
image = ax.imshow(
    normalized_image,
    origin="lower",
    cmap="RdBu_r",
    norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
    interpolation="nearest",
    rasterized=True,
)
ax.set_title("Spatial response relative to image median")
ax.set_xlabel("Detector column")
ax.set_ylabel("Detector row")
cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, extend="both")
cbar.set_label("Deviation from median (%)")

ax = axes[1, 0]
axis = np.arange(data.shape[0])
ax.plot(axis, (row_medians - 1.0) * 100.0, color="#c1121f", linewidth=0.8, label="row median")
ax.plot(axis, (column_medians - 1.0) * 100.0, color="#2f6690", linewidth=0.8, label="column median")
ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
ax.set_title("Detector-scale illumination structure")
ax.set_xlabel("Row or column index")
ax.set_ylabel("Median deviation (%)")
ax.legend(frameon=False)

ax = axes[1, 1]
ax.hist(normalized_values, bins=250, range=(0.90, 1.10), color="#2f6690", alpha=0.9)
ax.axvline(1.0, color="#c1121f", linestyle="--", linewidth=1.5, label="image median")
ax.set_yscale("log")
ax.set_title("Normalized pixel distribution")
ax.set_xlabel("Data / image median")
ax.set_ylabel("Pixels (log scale)")
ax.legend(frameon=False)
ax.text(
    0.03,
    0.97,
    f"median = {median:.3f}\n"
    f"valid pixels = {int(valid.sum()):,}\n"
    f"nonzero DQ = {dq_count:,}\n"
    "flat-field step = SKIPPED",
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)

fig.savefig(OUTPUT_PATH, dpi=180)
print(f"Wrote {OUTPUT_PATH}")
