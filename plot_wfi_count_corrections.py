"""Plot raw and count-corrected WFI tile count means."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fit_wfi_tmm_numpyro import apply_fixed_corrections, load_tile_statistics


EVENT_FIELDS = {
    14: ("ACM deployment", "reference_acm_step_counts"),
    35: ("pre-reset ACM", "reference_pre_reset_acm_step_counts"),
    39: ("additional ACM", "reference_additional_acm_step_counts"),
}


def load_event_steps(path, data):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No correction rows found in {path}")
    required = {"tile_id", "group", "count_log_correction", "log_gain_correction"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Correction table is missing columns: {sorted(missing)}")

    group_to_led = dict(zip(data["groups"], data["led"]))
    group_to_repeat = dict(zip(data["groups"], data["repeat"]))
    group_by_led_repeat = {
        (int(led), int(repeat)): int(group)
        for group, led, repeat in zip(data["groups"], data["led"], data["repeat"])
    }
    step_logs = {
        (int(row["tile_id"]), int(row["group"])): float(row["count_log_correction"])
        - float(row["log_gain_correction"])
        for row in rows
    }
    event_steps = {}
    for repeat, (label, _) in EVENT_FIELDS.items():
        percentages = []
        for (tile_id, group), current_step_log in step_logs.items():
            if group_to_repeat.get(group) != repeat:
                continue
            led = int(group_to_led[group])
            previous_group = group_by_led_repeat[(led, repeat - 1)]
            previous_step_log = step_logs[(tile_id, previous_group)]
            percentages.append(
                100.0 * np.expm1(current_step_log - previous_step_log)
            )
        if not percentages:
            continue
        event_steps[repeat] = (label, float(np.median(percentages)))
    return event_steps


def plot_counts(output, data, corrected, event_steps=None):
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    repeats = np.asarray(data["repeat"], dtype=int)
    raw = np.asarray(data["sum_current_counts"], dtype=float) / data["counts"]
    fixed = (
        np.asarray(corrected["sum_current_counts"], dtype=float)
        / data["counts"]
    )
    detector = str(np.asarray(data.get("sca_names", ["WFI"])).reshape(-1)[0])
    figure, axes = plt.subplots(
        2, 3, figsize=(17, 10), sharex=True, sharey=True,
    )
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.10, top=0.84,
                           wspace=0.08, hspace=0.16)
    event_steps = event_steps or {}
    for axis, led in zip(axes.flat, range(1, 7)):
        groups = np.flatnonzero(leds == led)
        x = days[groups]
        raw_values = raw[:, groups]
        fixed_values = fixed[:, groups]
        for tile_index in range(raw_values.shape[0]):
            axis.plot(x, raw_values[tile_index], color="0.65", alpha=0.16, linewidth=0.45)
            axis.plot(
                x,
                fixed_values[tile_index],
                color="tab:blue",
                alpha=0.16,
                linewidth=0.45,
            )
        fixed_quantiles = np.percentile(fixed_values, [16, 84], axis=0)
        axis.fill_between(
            x,
            fixed_quantiles[0],
            fixed_quantiles[1],
            color="tab:blue",
            alpha=0.2,
            label="after fixed correction 16th-84th percentile",
        )
        axis.plot(x, np.median(raw_values, axis=0), color="0.25", linewidth=1.4, label="raw median")
        axis.plot(
            x,
            np.median(fixed_values, axis=0),
            color="tab:blue",
            linewidth=1.8,
            marker="o",
            markersize=2.2,
            label="after fixed correction median",
        )
        axis.plot(
            x,
            np.median(raw_values, axis=0),
            color="0.25",
            linewidth=0.0,
            marker="o",
            markersize=2.0,
        )
        for event_repeat, (label, _) in event_steps.items():
            event_indices = groups[repeats[groups] == event_repeat]
            if event_indices.size == 0:
                continue
            event_index = int(event_indices[0])
            local_index = int(np.flatnonzero(groups == event_index)[0])
            event_raw = float(np.median(raw_values[:, local_index]))
            event_fixed = float(np.median(fixed_values[:, local_index]))
            event_day = days[event_index]
            axis.axvline(
                event_day,
                color="tab:red",
                alpha=0.28,
                linewidth=0.9,
                linestyle="--",
            )
            axis.annotate(
                f"r{event_repeat}: {event_raw:.0f} -> {event_fixed:.0f}",
                xy=(event_day, event_fixed),
                xytext=(3, 7 if event_repeat % 2 else -17),
                textcoords="offset points",
                fontsize=6.5,
                color="tab:red",
                rotation=90,
                va="bottom" if event_repeat % 2 else "top",
            )
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=7, loc="best")
    for axis in axes[-1]:
        axis.set_xlabel("Days since first exposure")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean current counts")
    step_note = ""
    if event_steps:
        step_note = "Median applied ACM corrections: " + "; ".join(
            f"r{repeat} {label} {step:+.2f}%"
            for repeat, (label, step) in event_steps.items()
        )
    figure.suptitle(
        f"{detector} tile-count medians before and after fixed count corrections (gain + ACM)",
        y=0.985,
    )
    figure.text(
        0.5,
        0.942,
        "Red labels show raw-median -> corrected-median counts at each event.",
        ha="center",
        va="center",
        fontsize=8,
    )
    if step_note:
        figure.text(
            0.5,
            0.915,
            step_note,
            ha="center",
            va="center",
            fontsize=8,
        )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--additional-acm-step-repeat", type=int, default=None)
    parser.add_argument("--pre-reset-acm-step-repeat", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_tile_statistics(
        args.statistics,
        additional_acm_step_repeat=args.additional_acm_step_repeat,
        pre_reset_acm_step_repeat=args.pre_reset_acm_step_repeat,
    )
    corrected = apply_fixed_corrections(data, args.corrections)
    if corrected.get("fixed_correction_space") != "counts":
        raise ValueError("The correction table must be marked as count-space")
    plot_counts(args.output, data, corrected, load_event_steps(args.corrections, data))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
