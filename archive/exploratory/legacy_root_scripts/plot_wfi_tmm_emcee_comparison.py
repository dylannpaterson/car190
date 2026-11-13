"""Plot and compare free-rate and fixed-rate WFI TMM EMCEE chains."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


FOCUSED_LABELS = (
    "Initial ice (nm)",
    "Decon2 ice (nm)",
    "Incidence angle (deg)",
    "Constant rate (nm/day)",
    "Post-Decon2 offset",
    "HgCdTe n log",
    "HgCdTe k log",
)
OPTICAL_COLUMNS = (0, 1, 2, 4, 5, 6)
OPTICAL_LABELS = tuple(FOCUSED_LABELS[index] for index in OPTICAL_COLUMNS)
FREE_COLOR = "#0072B2"
FIXED_COLOR = "#D55E00"


def constraint_label(run):
    constraints = []
    if run["is_fixed_angle"]:
        constraints.append(f"angle={run['fixed_angle']:.6g} deg")
    if run["is_fixed"]:
        constraints.append(f"rate={run['fixed_rate']:.6g} nm/day")
    return ", ".join(constraints) if constraints else "free angle and free rate"


def comparison_columns(fixed):
    columns = list(OPTICAL_COLUMNS)
    if fixed["is_fixed_angle"]:
        columns.remove(2)
    return tuple(columns)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--free-dir",
        type=Path,
        default=root / "20261012/wfi01_center_tile136_constant_rate_emcee_warmup1000_steps5000",
    )
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        default=root / "20261012/wfi01_center_tile136_fixed_rate_10p729_emcee_warmup1000_steps5000",
    )
    parser.add_argument("--label", default="WFI01 tile 136")
    parser.add_argument("--burn", type=int, default=500)
    return parser.parse_args()


def _scalar(archive, name, default=np.nan):
    if name not in archive:
        return float(default)
    return float(np.asarray(archive[name]).reshape(-1)[0])


def load_run(path):
    with np.load(path / "emcee_chain.npz", allow_pickle=False) as archive:
        chain = np.asarray(archive["chain"], dtype=float)
        log_probability = np.asarray(archive["log_probability"], dtype=float)
        acceptance = np.asarray(archive["acceptance_fraction"], dtype=float)
        fixed_rate = _scalar(archive, "fixed_rate_nm_day")
        fixed_angle = _scalar(archive, "fixed_incidence_angle_deg")
    warmup_path = path / "emcee_warmup_chain.npz"
    if warmup_path.is_file():
        with np.load(warmup_path, allow_pickle=False) as archive:
            warmup_chain = np.asarray(archive["chain"], dtype=float)
            warmup_log_probability = np.asarray(
                archive["log_probability"], dtype=float
            )
    else:
        warmup_chain = np.empty((0, chain.shape[1], chain.shape[2]))
        warmup_log_probability = np.empty((0, chain.shape[1]))
    if chain.ndim != 3 or log_probability.shape != chain.shape[:2]:
        raise ValueError(f"Invalid chain shape in {path}")
    is_fixed = np.isfinite(fixed_rate)
    is_fixed_angle = np.isfinite(fixed_angle)
    expected_dimensions = 13 - int(is_fixed) - int(is_fixed_angle)
    if chain.shape[-1] != expected_dimensions:
        raise ValueError(
            f"Expected {expected_dimensions} parameters in {path}, got {chain.shape[-1]}"
        )
    return {
        "path": path,
        "chain": chain,
        "log_probability": log_probability,
        "acceptance": acceptance,
        "warmup_chain": warmup_chain,
        "warmup_log_probability": warmup_log_probability,
        "fixed_rate": fixed_rate,
        "fixed_angle": fixed_angle,
        "is_fixed": is_fixed,
        "is_fixed_angle": is_fixed_angle,
    }


def focused_chain(run):
    chain = run["chain"]
    values = np.empty((chain.shape[0], chain.shape[1], len(FOCUSED_LABELS)))
    values[..., 0] = np.exp(chain[..., 6])
    values[..., 1] = np.exp(chain[..., 7])
    parameter_index = 8
    if run["is_fixed_angle"]:
        values[..., 2] = run["fixed_angle"]
    else:
        values[..., 2] = chain[..., parameter_index]
        parameter_index += 1
    if run["is_fixed"]:
        values[..., 3] = run["fixed_rate"]
    else:
        values[..., 3] = np.exp(chain[..., parameter_index])
        parameter_index += 1
    values[..., 4:] = chain[..., parameter_index : parameter_index + 3]
    return values


def focused_warmup_chain(run):
    warmup = run["warmup_chain"]
    if warmup.shape[0] == 0:
        return np.empty((0, warmup.shape[1], len(FOCUSED_LABELS)))
    copy = dict(run)
    copy["chain"] = warmup
    return focused_chain(copy)


def posterior_samples(run, burn):
    values = focused_chain(run)[burn:]
    return values.reshape(-1, values.shape[-1])


def save_trace_plot(run, path):
    production = focused_chain(run)
    warmup = focused_warmup_chain(run)
    steps, walkers, dimensions = production.shape
    if warmup.shape[0]:
        values = np.concatenate((warmup, production), axis=0)
        warmup_end = warmup.shape[0]
    else:
        values = production
        warmup_end = 0
    figure, axes = plt.subplots(
        dimensions,
        1,
        figsize=(15, 18),
        sharex=True,
        constrained_layout=True,
    )
    for index, (axis, label) in enumerate(zip(axes, FOCUSED_LABELS)):
        for walker in range(walkers):
            axis.plot(
                np.arange(values.shape[0]),
                values[:, walker, index],
                color=(
                    FIXED_COLOR
                    if run["is_fixed"] or run["is_fixed_angle"]
                    else FREE_COLOR
                ),
                alpha=0.12,
                linewidth=0.45,
            )
        axis.plot(
            np.arange(values.shape[0]),
            np.median(values[:, :, index], axis=1),
            color="black",
            linewidth=1.0,
        )
        if warmup_end:
            axis.axvline(warmup_end, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Step (warmup followed by saved production chain)")
    mode = constraint_label(run)
    figure.suptitle(
        f"{run['label']} {mode} EMCEE traces; {walkers} walkers; {steps} production steps"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_objective_plot(run, path):
    warmup = run["warmup_log_probability"]
    production = run["log_probability"]
    if warmup.shape[0]:
        values = np.concatenate((warmup, production), axis=0)
        warmup_end = warmup.shape[0]
    else:
        values = production
        warmup_end = 0
    objective = -values
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for walker in range(objective.shape[1]):
        axis.plot(
            np.arange(objective.shape[0]),
            objective[:, walker],
            color=(
                FIXED_COLOR
                if run["is_fixed"] or run["is_fixed_angle"]
                else FREE_COLOR
            ),
            alpha=0.12,
            linewidth=0.5,
        )
    axis.plot(np.median(objective, axis=1), color="black", label="Walker median")
    axis.plot(np.min(objective, axis=1), color="#009E73", label="Best walker")
    if warmup_end:
        axis.axvline(warmup_end, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Step (warmup followed by saved production chain)")
    axis.set_ylabel("Negative log posterior (lower is better)")
    axis.set_title(f"{run['label']} EMCEE objective trace")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def autocorrelation(values):
    import emcee

    try:
        return np.asarray(
            emcee.autocorr.integrated_time(values, quiet=True, tol=0), dtype=float
        )
    except Exception:
        return np.full(values.shape[-1], np.nan)


def save_individual_corner(run, samples, path, columns):
    import corner

    labels = tuple(FOCUSED_LABELS[index] for index in columns)
    optical = samples[:, list(columns)]
    figure = corner.corner(
        optical[: min(len(optical), 20000)],
        labels=labels,
        color=FIXED_COLOR if run["is_fixed"] or run["is_fixed_angle"] else FREE_COLOR,
        bins=35,
        smooth=1.0,
        show_titles=True,
        title_fmt=".4g",
        plot_datapoints=False,
        fill_contours=True,
    )
    figure.suptitle(
        f"{run['label']} optical posterior: " + constraint_label(run),
        y=1.02,
    )
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_comparison_marginals(free, fixed, path):
    figure, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    free_samples = free["samples"]
    fixed_samples = fixed["samples"]
    for index, (axis, label) in enumerate(zip(axes.flat, FOCUSED_LABELS)):
        if index == 3 and fixed["is_fixed"]:
            axis.hist(
                free_samples[:, index],
                bins=70,
                density=True,
                color=FREE_COLOR,
                alpha=0.55,
                label=constraint_label(free),
            )
            axis.axvline(
                fixed["fixed_rate"],
                color=FIXED_COLOR,
                linewidth=2.0,
                label=constraint_label(fixed),
            )
        else:
            axis.hist(
                free_samples[:, index],
                bins=70,
                density=True,
                color=FREE_COLOR,
                alpha=0.5,
                label=constraint_label(free),
            )
            axis.hist(
                fixed_samples[:, index],
                bins=70,
                density=True,
                color=FIXED_COLOR,
                alpha=0.5,
                label=constraint_label(fixed),
            )
        axis.set_xlabel(label)
        axis.set_ylabel("Posterior density")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    axes.flat[-1].axis("off")
    figure.suptitle(
        f"{free['label']} constrained versus unconstrained optical posteriors"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_comparison_correlations(free, fixed, path):
    columns = comparison_columns(fixed)
    labels = tuple(FOCUSED_LABELS[index] for index in columns)
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for axis, run, title in zip(
        axes,
        (free, fixed),
        (constraint_label(free), constraint_label(fixed)),
    ):
        correlation = np.corrcoef(run["samples"][:, list(columns)].T)
        image = axis.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axis.set_xticks(range(len(labels)), labels, rotation=65, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(title)
        for row in range(correlation.shape[0]):
            for column in range(correlation.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{correlation[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axes, label="Pearson correlation")
    figure.suptitle(f"{free['label']} optical posterior correlations")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_comparison_corner(free, fixed, path):
    import corner

    columns = comparison_columns(fixed)
    labels = tuple(FOCUSED_LABELS[index] for index in columns)
    free_optical = free["samples"][:, list(columns)]
    fixed_optical = fixed["samples"][:, list(columns)]
    limit = 15000
    figure = corner.corner(
        free_optical[:limit],
        labels=labels,
        color=FREE_COLOR,
        bins=35,
        smooth=1.0,
        plot_datapoints=False,
        fill_contours=False,
        show_titles=False,
    )
    corner.corner(
        fixed_optical[:limit],
        labels=labels,
        color=FIXED_COLOR,
        bins=35,
        smooth=1.0,
        plot_datapoints=False,
        fill_contours=False,
        show_titles=False,
        fig=figure,
    )
    figure.legend(
        handles=(
            Line2D(
                [0],
                [0],
                color=FREE_COLOR,
                lw=2,
                label=constraint_label(free),
            ),
            Line2D(
                [0],
                [0],
                color=FIXED_COLOR,
                lw=2,
                label=constraint_label(fixed),
            ),
        ),
        loc="upper right",
        frameon=False,
    )
    figure.suptitle(f"{free['label']} constrained optical posterior overlay", y=1.02)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_comparison_autocorrelation(free, fixed, burn, path):
    columns = comparison_columns(fixed)
    labels = tuple(FOCUSED_LABELS[index] for index in columns)
    optical_columns = list(columns)
    free_tau = autocorrelation(focused_chain(free)[burn:][..., optical_columns])
    fixed_tau = autocorrelation(focused_chain(fixed)[burn:][..., optical_columns])
    indices = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    axis.bar(
        indices - width / 2,
        free_tau,
        width,
        label=constraint_label(free),
        color=FREE_COLOR,
    )
    axis.bar(
        indices + width / 2,
        fixed_tau,
        width,
        label=constraint_label(fixed),
        color=FIXED_COLOR,
    )
    axis.set_xticks(indices, labels, rotation=45, ha="right")
    axis.set_ylabel("Integrated autocorrelation time")
    axis.set_title(
        f"{free['label']} autocorrelation times after {burn} production steps"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return free_tau, fixed_tau


def _interval(values):
    return np.percentile(values, [16, 50, 84])


def write_summary(free, fixed, free_tau, fixed_tau, burn, path):
    lines = [
        f"# {free['label']} EMCEE fixed/free comparison",
        "",
        f"- Free chain: `{free['path']}`.",
        f"- Fixed chain: `{fixed['path']}`.",
        f"- Production burn used for posterior summaries: `{burn}` steps.",
        f"- Reference constraint: `{constraint_label(free)}`.",
        f"- Constrained chain: `{constraint_label(fixed)}`.",
        f"- Reference mean acceptance: `{np.mean(free['acceptance']):.6g}`.",
        f"- Constrained mean acceptance: `{np.mean(fixed['acceptance']):.6g}`.",
        f"- Reference maximum log posterior: `{np.max(free['log_probability']):.10g}`.",
        f"- Constrained maximum log posterior: `{np.max(fixed['log_probability']):.10g}`.",
        "",
        "## Focused posterior intervals",
        "",
    ]
    for label, free_column, fixed_column in zip(
        FOCUSED_LABELS, free["samples"].T, fixed["samples"].T
    ):
        free_interval = _interval(free_column)
        fixed_interval = _interval(fixed_column)
        lines.append(
            f"- {label}: reference `{free_interval[1]:.8g}` "
            f"[{free_interval[0]:.8g}, {free_interval[2]:.8g}]; "
            f"constrained `{fixed_interval[1]:.8g}` "
            f"[{fixed_interval[0]:.8g}, {fixed_interval[2]:.8g}]."
    )
    lines.extend(("", "## Optical autocorrelation times", ""))
    columns = comparison_columns(fixed)
    labels = tuple(FOCUSED_LABELS[index] for index in columns)
    for label, free_value, fixed_value in zip(labels, free_tau, fixed_tau):
        lines.append(
            f"- {label}: reference `{free_value:.8g}`, "
            f"constrained `{fixed_value:.8g}`."
        )
    lines.extend(("", "## Strongest optical correlations", ""))
    correlation_columns = list(columns)
    for run, title in ((free, "reference"), (fixed, "constrained")):
        correlation = np.corrcoef(run["samples"][:, correlation_columns].T)
        pairs = []
        for row in range(len(labels)):
            for column in range(row):
                pairs.append(
                    (
                        abs(correlation[row, column]),
                        correlation[row, column],
                        labels[row],
                        labels[column],
                    )
                )
        lines.append(f"### {title.capitalize()} chain")
        lines.append("")
        for _, value, first, second in sorted(pairs, reverse=True)[:5]:
            lines.append(f"- `{first}` with `{second}`: `r={value:.6f}`.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    free = load_run(args.free_dir)
    fixed = load_run(args.fixed_dir)
    free["label"] = args.label
    fixed["label"] = args.label
    if fixed["is_fixed_angle"] and free["is_fixed_angle"]:
        if not np.isclose(fixed["fixed_angle"], free["fixed_angle"]):
            raise ValueError("Reference and constrained angle values differ")
    if fixed["is_fixed"] and free["is_fixed"]:
        if not np.isclose(fixed["fixed_rate"], free["fixed_rate"]):
            raise ValueError("Reference and constrained rate values differ")
    if not fixed["is_fixed"] and not fixed["is_fixed_angle"]:
        raise ValueError("The comparison chain must fix rate or angle")
    for run in (free, fixed):
        run["samples"] = posterior_samples(run, args.burn)
        plot_dir = run["path"] / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        individual_columns = list(OPTICAL_COLUMNS)
        if run["is_fixed_angle"]:
            individual_columns.remove(2)
        save_trace_plot(run, plot_dir / "chain_traces_warmup_and_production.png")
        save_objective_plot(run, plot_dir / "objective_trace_warmup_and_production.png")
        save_individual_corner(
            run,
            run["samples"],
            plot_dir
            / f"corner_ice_optical_{len(individual_columns)}_parameter.png",
            tuple(individual_columns),
        )
    comparison_dir = fixed["path"] / "plots"
    save_comparison_marginals(
        free,
        fixed,
        comparison_dir / "fixed_free_optical_marginals.png",
    )
    save_comparison_correlations(
        free,
        fixed,
        comparison_dir / "fixed_free_optical_correlations.png",
    )
    save_comparison_corner(
        free,
        fixed,
        comparison_dir
        / f"fixed_free_corner_ice_optical_{len(comparison_columns(fixed))}_parameter.png",
    )
    free_tau, fixed_tau = save_comparison_autocorrelation(
        free,
        fixed,
        args.burn,
        comparison_dir / "fixed_free_autocorrelation_times.png",
    )
    write_summary(
        free,
        fixed,
        free_tau,
        fixed_tau,
        args.burn,
        comparison_dir / "fixed_free_comparison_summary.md",
    )
    print(f"Wrote plots under {comparison_dir}")


if __name__ == "__main__":
    main()
