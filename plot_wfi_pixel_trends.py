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
QUANTILE_PATH = ROOT / "wfi01_pixel_trend_quantiles.csv"
PLOT_PATH = ROOT / "wfi01_pixel_trends_led_candlestick.png"

RNG = np.random.default_rng(1039)
CANDIDATE_PIXELS = 5000
PLOT_PIXELS_PER_LED = 800


def parse_mast_time(value):
    value = str(value).rstrip("Z")
    date_part, separator, fraction = value.partition(".")
    result = datetime.fromisoformat(date_part)
    if separator:
        result += timedelta(microseconds=int((fraction + "000000")[:6]))
    return result


def read_sampled_pixel_series(rows):
    rows = sorted(rows, key=lambda row: row["exposure_start_time"])
    first_path = DATA_DIR / f'{rows[0]["fileSetName"]}_cal.asdf'
    with asdf.open(first_path, lazy_load=True, ignore_unrecognized_tag=True) as af:
        first_data = np.asarray(af.tree["roman"]["data"])
        first_dq = np.asarray(af.tree["roman"]["dq"])

    n_pixels = first_data.size
    candidates = RNG.choice(n_pixels, size=min(CANDIDATE_PIXELS, n_pixels), replace=False)
    sampled_data = np.empty((len(rows), candidates.size), dtype=np.float32)
    stable = np.ones(candidates.size, dtype=bool)

    for index, row in enumerate(rows):
        path = DATA_DIR / f'{row["fileSetName"]}_cal.asdf'
        with asdf.open(path, lazy_load=True, ignore_unrecognized_tag=True) as af:
            model = af.tree["roman"]
            data = np.asarray(model["data"])
            dq = np.asarray(model["dq"])
        sampled_data[index] = data.flat[candidates]
        stable &= np.isfinite(sampled_data[index]) & (dq.flat[candidates] == 0)

    sampled_data = sampled_data[:, stable]
    if sampled_data.shape[1] < PLOT_PIXELS_PER_LED:
        raise RuntimeError(
            f"Only {sampled_data.shape[1]} stable sample pixels available; "
            f"need {PLOT_PIXELS_PER_LED}"
        )

    medians = np.median(sampled_data, axis=1)
    raw_change = (sampled_data / sampled_data[0] - 1.0) * 100.0
    frame_normalized = sampled_data / medians[:, None]
    normalized_change = (frame_normalized / frame_normalized[0] - 1.0) * 100.0

    plot_indices = RNG.choice(
        sampled_data.shape[1], size=min(PLOT_PIXELS_PER_LED, sampled_data.shape[1]), replace=False
    )
    start = parse_mast_time(rows[0]["exposure_start_time"])
    hours = np.asarray(
        [
            (parse_mast_time(row["exposure_start_time"]) - start).total_seconds() / 3600
            for row in rows
        ]
    )
    return {
        "rows": rows,
        "hours": hours,
        "medians": medians,
        "raw_change": raw_change,
        "normalized_change": normalized_change,
        "plot_indices": plot_indices,
        "stable_count": sampled_data.shape[1],
    }


with INDEX_PATH.open(newline="") as handle:
    indexed_rows = list(csv.DictReader(handle))

results = []
for led in sorted({int(row["led"]) for row in indexed_rows}):
    rows = [row for row in indexed_rows if int(row["led"]) == led]
    result = read_sampled_pixel_series(rows)
    result["led"] = led
    results.append(result)
    print(f"LED {led}: {result['stable_count']} stable sampled pixels")

with QUANTILE_PATH.open("w", newline="") as handle:
    fieldnames = [
        "led",
        "repeat",
        "time_hours_since_first",
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
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        for repeat, hours in enumerate(result["hours"]):
            raw_q = np.percentile(result["raw_change"][repeat], [5, 25, 50, 75, 95])
            normalized_q = np.percentile(result["normalized_change"][repeat], [5, 25, 50, 75, 95])
            writer.writerow(
                {
                    "led": result["led"],
                    "repeat": repeat,
                    "time_hours_since_first": f"{hours:.6f}",
                    "raw_p05": f"{raw_q[0]:.6f}",
                    "raw_p25": f"{raw_q[1]:.6f}",
                    "raw_median": f"{raw_q[2]:.6f}",
                    "raw_p75": f"{raw_q[3]:.6f}",
                    "raw_p95": f"{raw_q[4]:.6f}",
                    "normalized_p05": f"{normalized_q[0]:.6f}",
                    "normalized_p25": f"{normalized_q[1]:.6f}",
                    "normalized_median": f"{normalized_q[2]:.6f}",
                    "normalized_p75": f"{normalized_q[3]:.6f}",
                    "normalized_p95": f"{normalized_q[4]:.6f}",
                }
            )

colors = {led: plt.get_cmap("tab10")(led - 1) for led in range(1, 7)}
fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)
fig.suptitle(
    "WFI01 RCS-flat candidate: sampled pixel trends by LED\n"
    "Each trace is one stable detector pixel; colors identify LED channel",
    fontsize=16,
)

raw_all = np.concatenate(
    [result["raw_change"][:, result["plot_indices"]].ravel() for result in results]
)
normalized_all = np.concatenate(
    [result["normalized_change"][:, result["plot_indices"]].ravel() for result in results]
)
raw_limits = np.percentile(raw_all, [0.5, 99.5])
normalized_limits = np.percentile(normalized_all, [0.5, 99.5])

ax = axes[0, 0]
for result in results:
    led = result["led"]
    values = result["raw_change"][:, result["plot_indices"]]
    for pixel in range(values.shape[1]):
        ax.plot(result["hours"], values[:, pixel], color=colors[led], alpha=0.025, linewidth=0.35)
    ax.plot([], [], color=colors[led], linewidth=2.0, label=f"LED {led}")
ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
ax.set_ylim(raw_limits)
ax.set_title("Raw per-pixel change from first exposure")
ax.set_xlabel("Hours since first exposure in channel")
ax.set_ylabel("Pixel change (%)")
ax.legend(frameon=False, ncol=3)

ax = axes[0, 1]
for result in results:
    led = result["led"]
    values = result["normalized_change"][:, result["plot_indices"]]
    for pixel in range(values.shape[1]):
        ax.plot(result["hours"], values[:, pixel], color=colors[led], alpha=0.025, linewidth=0.35)
    ax.plot([], [], color=colors[led], linewidth=2.0, label=f"LED {led}")
ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
ax.set_ylim(normalized_limits)
ax.set_title("Per-pixel change after frame-median normalization")
ax.set_xlabel("Hours since first exposure in channel")
ax.set_ylabel("Pixel change (%)")
ax.legend(frameon=False, ncol=3)


def draw_candlesticks(axis, key, title, ylabel):
    for result in results:
        led = result["led"]
        values = result[key]
        offset = (led - 3.5) * 0.11
        for repeat in range(values.shape[0]):
            q05, q25, median, q75, q95 = np.percentile(values[repeat], [5, 25, 50, 75, 95])
            x = repeat + offset
            color = colors[led]
            axis.vlines(x, q05, q95, color=color, linewidth=1.0)
            axis.vlines(x, q25, q75, color=color, linewidth=7.0, alpha=0.9)
            axis.plot(x, median, marker="_", markersize=9, color="black", markeredgewidth=1.2)
        axis.plot([], [], color=colors[led], linewidth=5.0, label=f"LED {led}")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Repeat number")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(13))
    axis.legend(frameon=False, ncol=3)


draw_candlesticks(
    axes[1, 0],
    "raw_change",
    "Raw pixel-change distribution at each repeat",
    "Pixel change (%)",
)
draw_candlesticks(
    axes[1, 1],
    "normalized_change",
    "Frame-normalized pixel-change distribution at each repeat",
    "Pixel change (%)",
)

fig.text(
    0.5,
    0.005,
    "Top: 800 randomly selected stable pixels per LED, alpha=0.025. "
    "Bottom: whisker=5th-95th percentile, body=25th-75th percentile, black tick=median; "
    "quantiles use 5,000 sampled stable pixels per LED.",
    ha="center",
    fontsize=10,
)
fig.savefig(PLOT_PATH, dpi=180)
plt.close(fig)

print(f"Wrote {QUANTILE_PATH}")
print(f"Wrote {PLOT_PATH}")
