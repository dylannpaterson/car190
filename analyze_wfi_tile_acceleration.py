"""Explore accumulation-rate acceleration from a production tile fit.

This is deliberately a post-processing diagnostic. It does not alter the
production likelihood or refit the TMM model. The production trajectory is
piecewise linear in log(rate), so its native rate derivative is also reported
alongside separately smoothed log(rate) splines.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, make_smoothing_spline
from scipy.integrate import cumulative_trapezoid

from fit_wfi_tmm_production import load_tile_statistics


SMOOTHING_FACTORS = (0.0, 0.5, 1.0, 4.0)
SMOOTHING_LABELS = {
    0.0: "interpolating",
    0.5: "light",
    1.0: "moderate",
    4.0: "strong",
}
NATURAL_SPLINE_CONFIGS = (
    (None, "GCV"),
    (1.0, "lambda=1"),
    (10000.0, "lambda=1e4"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-points", type=int, default=600)
    return parser.parse_args()


def load_node_fit(fit_dir):
    with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        rates = np.asarray(samples["node_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        thickness = np.asarray(samples["np_node_thickness"], dtype=float).reshape(-1)
    return rates, thickness


def native_acceleration(times, node_days, log_rates):
    """Evaluate d(rate)/dt for the production piecewise-linear log-rate model."""
    times = np.asarray(times, dtype=float)
    interval = np.searchsorted(node_days, times, side="right") - 1
    interval = np.clip(interval, 0, len(node_days) - 2)
    log_rate = np.interp(times, node_days, log_rates)
    slopes = np.diff(log_rates) / np.diff(node_days)
    return np.exp(log_rate) * slopes[interval]


def segment_spline(node_days, node_rates, node_thickness, dense_points):
    log_rates = np.log(node_rates)
    linear = np.polyval(np.polyfit(node_days, log_rates, 1), node_days)
    residual_variance = max(float(np.var(log_rates - linear)), 1.0e-12)
    dense_days = np.linspace(node_days[0], node_days[-1], dense_points)
    results = []
    for factor in SMOOTHING_FACTORS:
        smoothing = factor * len(node_days) * residual_variance
        spline = UnivariateSpline(
            node_days,
            log_rates,
            k=min(3, len(node_days) - 1),
            s=smoothing,
        )
        dense_log_rate = spline(dense_days)
        dense_rate = np.exp(dense_log_rate)
        dense_acceleration = dense_rate * spline.derivative(1)(dense_days)
        dense_thickness = node_thickness[0] + cumulative_trapezoid(
            dense_rate, dense_days, initial=0.0
        )
        results.append(
            {
                "factor": factor,
                "label": SMOOTHING_LABELS[factor],
                "smoothing": smoothing,
                "spline": spline,
                "days": dense_days,
                "rate": dense_rate,
                "acceleration": dense_acceleration,
                "thickness": dense_thickness,
            }
        )
    return results


def segment_thickness_spline(node_days, node_thickness, dense_points):
    linear = np.polyval(np.polyfit(node_days, node_thickness, 1), node_days)
    residual_variance = max(float(np.var(node_thickness - linear)), 1.0e-12)
    dense_days = np.linspace(node_days[0], node_days[-1], dense_points)
    results = []
    for factor in SMOOTHING_FACTORS:
        smoothing = factor * len(node_days) * residual_variance
        spline = UnivariateSpline(
            node_days,
            node_thickness,
            k=min(3, len(node_days) - 1),
            s=smoothing,
        )
        results.append(
            {
                "factor": factor,
                "label": SMOOTHING_LABELS[factor],
                "smoothing": smoothing,
                "spline": spline,
                "days": dense_days,
                "thickness": spline(dense_days),
                "rate": spline.derivative(1)(dense_days),
                "acceleration": spline.derivative(2)(dense_days),
            }
        )
    return results


def segment_natural_log_rate_splines(node_days, node_rates, dense_points):
    dense_days = np.linspace(node_days[0], node_days[-1], dense_points)
    log_rates = np.log(node_rates)
    results = []
    for smoothing, label in NATURAL_SPLINE_CONFIGS:
        spline = make_smoothing_spline(node_days, log_rates, lam=smoothing)
        dense_log_rate = spline(dense_days)
        dense_rate = np.exp(dense_log_rate)
        dense_acceleration = dense_rate * spline.derivative(1)(dense_days)
        results.append(
            {
                "factor": label,
                "label": label,
                "smoothing": "" if smoothing is None else smoothing,
                "spline": spline,
                "days": dense_days,
                "rate": dense_rate,
                "acceleration": dense_acceleration,
            }
        )
    return results


def write_summary(csv_path, markdown_path, summaries, reset_day, gap):
    fields = [
        "segment",
        "method",
        "smoothing_factor",
        "smoothing_residual_sum",
        "start_rate_nm_day",
        "end_rate_nm_day",
        "median_acceleration_nm_day2",
        "min_acceleration_nm_day2",
        "max_acceleration_nm_day2",
        "fraction_negative_acceleration",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    lines = [
        "# Tile acceleration exploratory analysis",
        "",
        f"- Physical reset day: `{reset_day:.15f}`.",
        f"- Gap from the last pre-reset node to the first post-reset node: `{gap:.8g}` days.",
        "- The reset is excluded from cross-boundary derivative interpretation.",
        "- Spline smoothing is applied to log(rate), then rate and acceleration are derived analytically.",
        "- Acceleration units are nm/day^2; fractional negative acceleration is evaluated on the dense grid.",
        "",
    ]
    for row in summaries:
        lines.append(
            "- {segment} / {method}: rate `{start_rate_nm_day:.6g}` -> "
            "`{end_rate_nm_day:.6g}` nm/day; median acceleration "
            "`{median_acceleration_nm_day2:.6g}` nm/day^2; negative fraction "
            "`{fraction_negative_acceleration:.3f}`.".format(**row)
        )
    markdown_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.dense_points < 50:
        raise ValueError("--dense-points must be at least 50")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_tile_statistics(args.statistics)
    node_days = np.asarray(data["node_days"], dtype=float)
    reset_day = float(data["reset_day"])
    node_rates, node_thickness = load_node_fit(args.fit_dir)
    if len(node_days) != len(node_rates) or len(node_days) != len(node_thickness):
        raise ValueError("Node metadata and posterior trajectory lengths do not match")

    pre = node_days < reset_day
    post = node_days >= reset_day
    if np.count_nonzero(pre) < 4 or np.count_nonzero(post) < 4:
        raise ValueError("Both reset segments need at least four trajectory nodes")
    gap = float(node_days[post][0] - node_days[pre][-1])

    segment_inputs = {
        "pre-reset": (node_days[pre], node_rates[pre], node_thickness[pre]),
        "post-reset": (node_days[post], node_rates[post], node_thickness[post]),
    }
    spline_results = {}
    thickness_spline_results = {}
    natural_spline_results = {}
    summaries = []
    for segment, (days, rates, thickness) in segment_inputs.items():
        results = segment_spline(days, rates, thickness, args.dense_points)
        spline_results[segment] = results
        for result in results:
            acceleration = result["acceleration"]
            summaries.append(
                {
                    "segment": segment,
                    "method": "log-rate spline",
                    "smoothing_factor": result["factor"],
                    "smoothing_residual_sum": result["smoothing"],
                    "start_rate_nm_day": result["rate"][0],
                    "end_rate_nm_day": result["rate"][-1],
                    "median_acceleration_nm_day2": np.median(acceleration),
                    "min_acceleration_nm_day2": np.min(acceleration),
                    "max_acceleration_nm_day2": np.max(acceleration),
                    "fraction_negative_acceleration": np.mean(acceleration < 0.0),
                }
            )
        thickness_results = segment_thickness_spline(
            days, thickness, args.dense_points
        )
        thickness_spline_results[segment] = thickness_results
        for result in thickness_results:
            acceleration = result["acceleration"]
            summaries.append(
                {
                    "segment": segment,
                    "method": "thickness spline",
                    "smoothing_factor": result["factor"],
                    "smoothing_residual_sum": result["smoothing"],
                    "start_rate_nm_day": result["rate"][0],
                    "end_rate_nm_day": result["rate"][-1],
                    "median_acceleration_nm_day2": np.median(acceleration),
                    "min_acceleration_nm_day2": np.min(acceleration),
                    "max_acceleration_nm_day2": np.max(acceleration),
                    "fraction_negative_acceleration": np.mean(acceleration < 0.0),
                }
            )
        natural_results = segment_natural_log_rate_splines(
            days, rates, args.dense_points
        )
        natural_spline_results[segment] = natural_results
        for result in natural_results:
            acceleration = result["acceleration"]
            summaries.append(
                {
                    "segment": segment,
                    "method": "natural log-rate spline",
                    "smoothing_factor": result["factor"],
                    "smoothing_residual_sum": result["smoothing"],
                    "start_rate_nm_day": result["rate"][0],
                    "end_rate_nm_day": result["rate"][-1],
                    "median_acceleration_nm_day2": np.median(acceleration),
                    "min_acceleration_nm_day2": np.min(acceleration),
                    "max_acceleration_nm_day2": np.max(acceleration),
                    "fraction_negative_acceleration": np.mean(acceleration < 0.0),
                }
            )

    for segment, (days, rates, _) in segment_inputs.items():
        dense_days = np.linspace(days[0], days[-1], args.dense_points)
        acceleration = native_acceleration(dense_days, days, np.log(rates))
        summaries.append(
            {
                "segment": segment,
                "method": "native piecewise log-rate",
                "smoothing_factor": "",
                "smoothing_residual_sum": "",
                "start_rate_nm_day": rates[0],
                "end_rate_nm_day": rates[-1],
                "median_acceleration_nm_day2": np.median(acceleration),
                "min_acceleration_nm_day2": np.min(acceleration),
                "max_acceleration_nm_day2": np.max(acceleration),
                "fraction_negative_acceleration": np.mean(acceleration < 0.0),
            }
        )

    write_summary(
        args.output_dir / "tile_acceleration_summary.csv",
        args.output_dir / "tile_acceleration_summary.md",
        summaries,
        reset_day,
        gap,
    )

    pre_days, pre_rates, pre_thickness = segment_inputs["pre-reset"]
    post_days, post_rates, post_thickness = segment_inputs["post-reset"]
    native_pre_days = np.linspace(pre_days[0], pre_days[-1], args.dense_points)
    native_post_days = np.linspace(post_days[0], post_days[-1], args.dense_points)
    native_pre_acceleration = native_acceleration(native_pre_days, pre_days, np.log(pre_rates))
    native_post_acceleration = native_acceleration(native_post_days, post_days, np.log(post_rates))

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    axes[0].scatter(pre_days, pre_thickness, color="#0072B2", s=20, label="Pre-reset TMM nodes")
    axes[0].scatter(post_days, post_thickness, color="#D55E00", s=20, label="Post-reset TMM nodes")
    for segment, results in spline_results.items():
        color = "#0072B2" if segment == "pre-reset" else "#D55E00"
        for result in results:
            alpha = 0.35 if result["factor"] != 1.0 else 0.95
            linewidth = 1.0 if result["factor"] != 1.0 else 2.0
            axes[0].plot(
                result["days"],
                result["thickness"],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                label=f"{segment} {result['label']} spline" if result["factor"] == 1.0 else None,
            )
    axes[0].set_ylabel("Thickness (nm)")
    axes[0].set_title("Fitted thickness and reset-separated smooth reconstructions")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].scatter(pre_days, pre_rates, color="#0072B2", s=20)
    axes[1].scatter(post_days, post_rates, color="#D55E00", s=20)
    for segment, results in spline_results.items():
        color = "#0072B2" if segment == "pre-reset" else "#D55E00"
        for result in results:
            alpha = 0.35 if result["factor"] != 1.0 else 0.95
            linewidth = 1.0 if result["factor"] != 1.0 else 2.0
            axes[1].plot(result["days"], result["rate"], color=color, alpha=alpha, linewidth=linewidth)
    axes[1].set_ylabel("Rate (nm/day)")
    axes[1].set_title("Accumulation rate: node values and log-rate splines")

    axes[2].plot(native_pre_days, native_pre_acceleration, color="#0072B2", alpha=0.5, linewidth=1.0, label="Native piecewise model")
    axes[2].plot(native_post_days, native_post_acceleration, color="#D55E00", alpha=0.5, linewidth=1.0)
    for segment, results in spline_results.items():
        color = "#0072B2" if segment == "pre-reset" else "#D55E00"
        for result in results:
            alpha = 0.35 if result["factor"] != 1.0 else 0.95
            linewidth = 1.0 if result["factor"] != 1.0 else 2.0
            label = f"{segment} {result['label']} spline" if result["factor"] == 1.0 else None
            axes[2].plot(result["days"], result["acceleration"], color=color, alpha=alpha, linewidth=linewidth, label=label)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Days since first exposure")
    axes[2].set_ylabel("d(rate)/dt (nm/day^2)")
    axes[2].set_title("Accumulation-rate acceleration; reset excluded")
    axes[2].legend(frameon=False, fontsize=8, ncol=2)

    for axis in axes:
        axis.axvline(reset_day, color="#555555", linestyle="--", linewidth=1.0, label="Physical reset" if axis is axes[0] else None)
        axis.grid(alpha=0.2)
    fig.suptitle("WFI01 tile 000 exploratory accumulation-rate analysis")
    fig.savefig(args.output_dir / "tile_acceleration_diagnostic.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    for segment, results in thickness_spline_results.items():
        color = "#0072B2" if segment == "pre-reset" else "#D55E00"
        for result in results:
            alpha = 0.35 if result["factor"] != 1.0 else 0.95
            linewidth = 1.0 if result["factor"] != 1.0 else 2.0
            label = f"{segment} {result['label']} spline" if result["factor"] == 1.0 else None
            axes[0].plot(result["days"], result["thickness"], color=color, alpha=alpha, linewidth=linewidth, label=label)
            axes[1].plot(result["days"], result["rate"], color=color, alpha=alpha, linewidth=linewidth, label=label)
            axes[2].plot(result["days"], result["acceleration"], color=color, alpha=alpha, linewidth=linewidth, label=label)
    axes[0].scatter(pre_days, pre_thickness, color="#0072B2", s=16)
    axes[0].scatter(post_days, post_thickness, color="#D55E00", s=16)
    axes[0].set_ylabel("Thickness (nm)")
    axes[0].set_title("Direct cubic smoothing splines fitted to TMM thickness nodes")
    axes[1].set_ylabel("d(thickness)/dt (nm/day)")
    axes[1].set_title("Rate obtained by differentiating the thickness spline")
    axes[2].set_ylabel("d2(thickness)/dt2 (nm/day^2)")
    axes[2].set_xlabel("Days since first exposure")
    axes[2].set_title("Acceleration obtained by differentiating the thickness spline")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes:
        axis.axvline(reset_day, color="#555555", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.2)
    fig.suptitle("WFI01 tile 000 thickness-spline sensitivity; reset-separated")
    fig.savefig(args.output_dir / "tile_thickness_spline_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for segment, results in natural_spline_results.items():
        color = "#0072B2" if segment == "pre-reset" else "#D55E00"
        for result in results:
            alpha = 0.45 if result["label"] != "lambda=1e4" else 0.95
            linewidth = 1.0 if result["label"] != "lambda=1e4" else 2.0
            label = f"{segment} {result['label']}" if result["label"] in ("GCV", "lambda=1e4") else None
            axes[0].plot(result["days"], result["rate"], color=color, alpha=alpha, linewidth=linewidth, label=label)
            axes[1].plot(result["days"], result["acceleration"], color=color, alpha=alpha, linewidth=linewidth, label=label)
    axes[0].scatter(pre_days, pre_rates, color="#0072B2", s=16)
    axes[0].scatter(post_days, post_rates, color="#D55E00", s=16)
    axes[0].set_ylabel("Rate (nm/day)")
    axes[0].set_title("Natural cubic smoothing splines on log(rate)")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Days since first exposure")
    axes[1].set_ylabel("d(rate)/dt (nm/day^2)")
    axes[1].set_title("Natural-spline acceleration")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes:
        axis.axvline(reset_day, color="#555555", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.2)
    fig.suptitle("WFI01 tile 000 natural-spline sensitivity; reset-separated")
    fig.savefig(args.output_dir / "tile_natural_spline_comparison.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {args.output_dir / 'tile_acceleration_diagnostic.png'}")
    print(f"Wrote {args.output_dir / 'tile_thickness_spline_comparison.png'}")
    print(f"Wrote {args.output_dir / 'tile_natural_spline_comparison.png'}")
    print(f"Wrote {args.output_dir / 'tile_acceleration_summary.csv'}")
    print(f"Wrote {args.output_dir / 'tile_acceleration_summary.md'}")


if __name__ == "__main__":
    main()
