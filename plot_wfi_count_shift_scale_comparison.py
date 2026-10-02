"""Plot comparable WFI01/WFI02 count-space shift and grey-scale fits."""

import argparse
import json
from pathlib import Path

import corner
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fit_wfi_count_shift_scale_mcmc import load_model_data


RESET_DAY = 15.064405023148149
PRE_COLOR = "tab:blue"
POST_COLOR = "tab:green"
DETECTOR_COLORS = {"WFI01": "tab:blue", "WFI02": "tab:orange"}


def load_posterior(path):
    with np.load(path, allow_pickle=False) as archive:
        chain = np.asarray(archive["chain"], dtype=float)
    if chain.ndim != 3 or chain.shape[-1] != 3:
        raise ValueError(f"Expected a (step, walker, 3) chain in {path}")
    flat = chain.reshape(-1, chain.shape[-1])
    return {
        "chain": chain,
        "delta_t": flat[:, 0],
        "grey": np.exp(flat[:, 1]),
        "sigma": np.exp(flat[:, 2]),
    }


def posterior_quantiles(values):
    return {
        str(level): float(np.quantile(values, level))
        for level in (0.025, 0.16, 0.5, 0.84, 0.975)
    }


def overlay_data(model, delta_t, grey):
    shift_days = float(delta_t - model["reset_day"])
    return [
        {
            "led": segment["led"],
            "pre_days": segment["pre_days"],
            "post_days": segment["post_days"] + shift_days,
            "pre_values": segment["pre_values"],
            "post_values": segment["post_values"] / grey,
        }
        for segment in model["segments"]
    ]


def draw_overlay_axis(axis, segment, show_legend=False):
    pre_days = segment["pre_days"]
    post_days = segment["post_days"]
    pre_values = segment["pre_values"]
    post_values = segment["post_values"]
    for tile_index in range(pre_values.shape[0]):
        axis.plot(
            pre_days,
            pre_values[tile_index],
            color=PRE_COLOR,
            alpha=0.12,
            linewidth=0.45,
        )
        axis.plot(
            post_days,
            post_values[tile_index],
            color=POST_COLOR,
            alpha=0.12,
            linewidth=0.45,
        )
    axis.plot(
        pre_days,
        np.median(pre_values, axis=0),
        color=PRE_COLOR,
        marker="o",
        linewidth=2.0,
        label="Pre-decon median",
    )
    axis.plot(
        post_days,
        np.median(post_values, axis=0),
        color=POST_COLOR,
        marker="o",
        linewidth=2.0,
        label="Post-decon / median grey scale",
    )
    axis.set_title(f"LED {segment['led']}")
    axis.set_xlabel("Time from t=0 (days)")
    axis.set_ylabel("Corrected current counts")
    axis.grid(alpha=0.2)
    if show_legend:
        axis.legend(frameon=False, loc="upper right")


def save_detector_overlay(output_dir, detector, model, posterior):
    median_delta_t = float(np.median(posterior["delta_t"]))
    median_grey = float(np.median(posterior["grey"]))
    segments = overlay_data(model, median_delta_t, median_grey)
    figure, axes = plt.subplots(2, 3, figsize=(20, 10.7), sharey=True)
    for index, segment in enumerate(segments):
        draw_overlay_axis(axes.flat[index], segment, show_legend=index == 0)
    figure.suptitle(
        f"{detector} final overlay: delta_t from t=0={median_delta_t:.5f} d, "
        f"grey change={100.0 * (median_grey - 1.0):+.4f}%"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = output_dir / f"{detector.lower()}_overlay_median_delta_t0_counts.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return path


def save_combined_overlay(output_dir, models, posteriors):
    figure, axes = plt.subplots(4, 3, figsize=(20, 20), sharey=True)
    for detector_index, detector in enumerate(("WFI01", "WFI02")):
        posterior = posteriors[detector]
        model = models[detector]
        median_delta_t = float(np.median(posterior["delta_t"]))
        median_grey = float(np.median(posterior["grey"]))
        segments = overlay_data(model, median_delta_t, median_grey)
        for segment_index, segment in enumerate(segments):
            axis = axes[2 * detector_index + segment_index // 3, segment_index % 3]
            draw_overlay_axis(axis, segment, show_legend=segment_index == 0)
            axis.set_title(f"{detector} LED {segment['led']}")
    figure.suptitle("WFI01/WFI02 final count-space shift/scale overlays")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path = output_dir / "wfi01_wfi02_overlay_median_delta_t0_counts.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return path


def save_posterior_comparison(output_dir, posteriors):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for detector in ("WFI01", "WFI02"):
        posterior = posteriors[detector]
        color = DETECTOR_COLORS[detector]
        label = f"{detector} median"
        axes[0].hist(
            posterior["delta_t"],
            bins=80,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=label,
        )
        axes[1].hist(
            100.0 * (posterior["grey"] - 1.0),
            bins=80,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=label,
        )
        step = max(1, len(posterior["delta_t"]) // 30000)
        axes[2].scatter(
            posterior["delta_t"][::step],
            100.0 * (posterior["grey"][::step] - 1.0),
            s=1,
            alpha=0.12,
            color=color,
            label=label,
        )
    axes[0].set_xlabel("delta_t from t=0 (days)")
    axes[0].set_ylabel("Posterior samples")
    axes[1].set_xlabel("Grey throughput change (%)")
    axes[1].set_ylabel("Posterior samples")
    axes[2].set_xlabel("delta_t from t=0 (days)")
    axes[2].set_ylabel("Grey change (%)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("WFI01/WFI02 posterior measured from t=0")
    figure.tight_layout()
    path = output_dir / "wfi01_wfi02_posterior_comparison_delta_t0.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def save_corner_comparison(output_dir, posteriors):
    values = {}
    for detector in ("WFI01", "WFI02"):
        posterior = posteriors[detector]
        values[detector] = np.column_stack((posterior["delta_t"], posterior["grey"]))
    figure = corner.corner(
        values["WFI01"],
        labels=["delta_t from t=0 (days)", "Grey scale"],
        color=DETECTOR_COLORS["WFI01"],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        plot_datapoints=False,
    )
    corner.corner(
        values["WFI02"],
        fig=figure,
        labels=["delta_t from t=0 (days)", "Grey scale"],
        color=DETECTOR_COLORS["WFI02"],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        plot_datapoints=False,
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color=DETECTOR_COLORS["WFI01"], label="WFI01"),
            Line2D([0], [0], color=DETECTOR_COLORS["WFI02"], label="WFI02"),
        ],
        loc="upper right",
        frameon=False,
    )
    figure.suptitle("WFI01/WFI02 MCMC posterior comparison", y=1.02)
    path = output_dir / "wfi01_wfi02_corner_comparison_delta_t0.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def save_trace_comparison(output_dir, posteriors):
    figure, axes = plt.subplots(3, 2, figsize=(18, 10), sharex="col")
    labels = ("delta_t from t=0 (days)", "Grey scale", "Sigma (counts)")
    for detector_index, detector in enumerate(("WFI01", "WFI02")):
        chain = posteriors[detector]["chain"]
        trace_values = (chain[:, :, 0], np.exp(chain[:, :, 1]), np.exp(chain[:, :, 2]))
        for row, (axis, values, label) in enumerate(zip(axes[:, detector_index], trace_values, labels)):
            axis.plot(values, color="tab:blue", alpha=0.08, linewidth=0.25)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(detector)
        axes[-1, detector_index].set_xlabel("Production step")
    figure.suptitle("WFI01/WFI02 MCMC traces measured from t=0")
    figure.tight_layout()
    path = output_dir / "wfi01_wfi02_traces_comparison_delta_t0.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def build_summary(models, posteriors, paths, output_dir):
    summary = {
        "reset_day": RESET_DAY,
        "overlay_definition": (
            "Raw statistics receive the base count-space correction table. "
            "Post-reset times are shifted by delta_t_from_t0 - reset_day, "
            "and post-reset values are divided by the posterior-median grey scale."
        ),
        "detectors": {},
        "plots": {name: str(path) for name, path in paths.items()},
    }
    for detector in ("WFI01", "WFI02"):
        posterior = posteriors[detector]
        summary["detectors"][detector] = {
            "chain_shape": list(posterior["chain"].shape),
            "delta_t_from_t0_days": posterior_quantiles(posterior["delta_t"]),
            "shift_days": posterior_quantiles(posterior["delta_t"] - RESET_DAY),
            "grey_scale": posterior_quantiles(posterior["grey"]),
            "grey_change_percent": posterior_quantiles(
                100.0 * (posterior["grey"] - 1.0)
            ),
            "sigma_counts": posterior_quantiles(posterior["sigma"]),
            "n_tiles": int(models[detector]["segments"][0]["pre_values"].shape[0]),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for detector in ("wfi01", "wfi02"):
        parser.add_argument(f"--{detector}-statistics", type=Path, required=True)
        parser.add_argument(f"--{detector}-corrections", type=Path, required=True)
        parser.add_argument(f"--{detector}-posterior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reset-day", type=float, default=RESET_DAY)
    parser.add_argument("--additional-acm-step-repeat", type=int, default=39)
    parser.add_argument("--pre-reset-acm-step-repeat", type=int, default=35)
    args = parser.parse_args()
    if args.reset_day != RESET_DAY:
        raise ValueError("This comparison uses the validated WFI01/WFI02 reset day")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    posteriors = {}
    for detector in ("wfi01", "wfi02"):
        prefix = detector
        models[detector.upper()] = load_model_data(
            getattr(args, f"{prefix}_statistics"),
            getattr(args, f"{prefix}_corrections"),
            args.reset_day,
            args.additional_acm_step_repeat,
            args.pre_reset_acm_step_repeat,
        )
        posteriors[detector.upper()] = load_posterior(
            getattr(args, f"{prefix}_posterior")
        )

    paths = {
        "wfi01_overlay": save_detector_overlay(
            output_dir, "WFI01", models["WFI01"], posteriors["WFI01"]
        ),
        "wfi02_overlay": save_detector_overlay(
            output_dir, "WFI02", models["WFI02"], posteriors["WFI02"]
        ),
        "combined_overlay": save_combined_overlay(output_dir, models, posteriors),
        "posterior_comparison": save_posterior_comparison(output_dir, posteriors),
        "corner_comparison": save_corner_comparison(output_dir, posteriors),
        "trace_comparison": save_trace_comparison(output_dir, posteriors),
    }
    summary = build_summary(models, posteriors, paths, output_dir)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    (output_dir / "summary.md").write_text(
        "# WFI01/WFI02 Count-Space Shift/Scale Comparison\n\n"
        "The overlays reproduce the validated WFI01 convention: pre-reset curves "
        "are plotted in blue, post-reset curves are shifted onto the pre-reset time "
        "axis and divided by the fitted grey scale in green. The input statistics "
        "are raw RRN statistics with the base count-space correction table.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
