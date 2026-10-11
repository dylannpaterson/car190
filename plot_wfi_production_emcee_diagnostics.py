"""Plot convergence diagnostics for the fixed production EMCEE chains."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_STATISTICS_DIR = ROOT / "20261008/corrected_statistics_program1063_4h"
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))
PARAMETER_LABELS = (
    "AR raw 0",
    "AR raw 1",
    "AR raw 2",
    "AR raw 3",
    "AR raw 4",
    "AR raw 5",
    "Initial thickness (nm)",
    "Reset thickness (nm)",
    "Incidence angle (deg)",
    "Constant rate (nm/day)",
    "Post-reset offset",
    "HgCdTe n log scale",
    "HgCdTe k log scale",
)


def load_chain(path):
    with np.load(path, allow_pickle=False) as archive:
        chain = np.asarray(archive["chain"], dtype=float)
        log_probability = np.asarray(archive["log_probability"], dtype=float)
        acceptance = np.asarray(archive["acceptance_fraction"], dtype=float)
    if chain.ndim != 3 or chain.shape[-1] != len(PARAMETER_LABELS):
        raise ValueError(f"Unexpected chain shape in {path}: {chain.shape}")
    if log_probability.shape != chain.shape[:2]:
        raise ValueError(f"Unexpected log-probability shape in {path}")
    return chain, log_probability, acceptance


def display_chain_values(chain):
    values = np.asarray(chain, dtype=float).copy()
    values[:, :, 6] = np.exp(values[:, :, 6])
    values[:, :, 7] = np.exp(values[:, :, 7])
    values[:, :, 9] = np.exp(values[:, :, 9])
    return values


def integrated_time(chain):
    try:
        import emcee

        return np.asarray(emcee.autocorr.integrated_time(chain, quiet=True, tol=0), dtype=float)
    except Exception:
        return np.full(chain.shape[-1], np.nan)


def plot_parameter_traces(path, detector, chain):
    values = display_chain_values(chain)
    steps, walkers, _ = values.shape
    figure, axes = plt.subplots(7, 2, figsize=(15, 20), sharex=True, constrained_layout=True)
    axes = axes.flat
    for parameter_index, (axis, label) in enumerate(zip(axes, PARAMETER_LABELS)):
        for walker in range(walkers):
            axis.plot(
                np.arange(steps),
                values[:, walker, parameter_index],
                color="#0072B2",
                alpha=0.13,
                linewidth=0.45,
            )
        axis.plot(
            np.arange(steps),
            np.median(values[:, :, parameter_index], axis=1),
            color="#D55E00",
            linewidth=1.2,
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-2].set_xlabel("EMCEE step")
    axes[-1].set_xlabel("EMCEE step")
    figure.suptitle(
        f"{detector} production EMCEE parameter traces\n"
        f"{walkers} walkers; {steps} steps; orange = per-step walker median",
        fontsize=14,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_objective_trace(path, detector, log_probability):
    objective = -np.asarray(log_probability, dtype=float)
    steps, walkers = objective.shape
    x = np.arange(steps)
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for walker in range(walkers):
        axis.plot(x, objective[:, walker], color="#0072B2", alpha=0.13, linewidth=0.5)
    axis.plot(x, np.median(objective, axis=1), color="#D55E00", linewidth=1.6, label="Walker median")
    axis.plot(x, np.min(objective, axis=1), color="black", linewidth=1.1, label="Best walker")
    axis.set_xlabel("EMCEE step")
    axis.set_ylabel("Negative log posterior (lower is better)")
    axis.set_title(f"{detector} production EMCEE objective traces")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_summary(path, detector, chain, log_probability, acceptance, tau):
    steps, walkers, dimensions = chain.shape
    tail = max(1, steps // 4)
    tail_log_probability = log_probability[-tail:]
    lines = [
        f"# {detector} production EMCEE diagnostics",
        "",
        f"- Chain shape: `{steps} steps x {walkers} walkers x {dimensions} parameters`.",
        f"- Finite log posterior fraction: `{np.isfinite(log_probability).mean():.6g}`.",
        f"- Acceptance fraction: mean `{np.mean(acceptance):.6g}`, median `{np.median(acceptance):.6g}`, min `{np.min(acceptance):.6g}`, max `{np.max(acceptance):.6g}`.",
        f"- Final-step best log posterior: `{np.max(log_probability[-1]):.10g}`.",
        f"- Overall best log posterior: `{np.max(log_probability):.10g}`.",
        f"- Final-quarter median log posterior: `{np.median(tail_log_probability):.10g}`.",
        f"- Final-quarter mean log posterior: `{np.mean(tail_log_probability):.10g}`.",
        f"- Integrated autocorrelation times: `{tau.tolist()}`.",
        f"- Maximum integrated autocorrelation time: `{np.nanmax(tau):.6g}`.",
        f"- Approximate steps per maximum autocorrelation time: `{steps / np.nanmax(tau):.6g}`.",
        "",
        "The chain is not treated as formally converged solely from this diagnostic. A rising best-posterior trace or fewer than roughly 50 autocorrelation times indicates that a longer or independently initialized chain is needed.",
        "",
    ]
    path.write_text("\n".join(lines))


def plot_objective_comparison(path, loaded):
    figure, axes = plt.subplots(len(loaded), 1, figsize=(13, 5 * len(loaded)), sharex=True, constrained_layout=True)
    if len(loaded) == 1:
        axes = [axes]
    for axis, (detector, (_chain, log_probability, _acceptance, _tau)) in zip(axes, loaded.items()):
        objective = -log_probability
        x = np.arange(objective.shape[0])
        axis.plot(x, np.median(objective, axis=1), label="Walker median")
        axis.plot(x, np.min(objective, axis=1), label="Best walker")
        axis.set_ylabel(f"{detector}\n-negative log posterior")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[-1].set_xlabel("EMCEE step")
    figure.suptitle("Production EMCEE objective comparison")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, default=DEFAULT_STATISTICS_DIR)
    parser.add_argument("--detector", action="append", choices=DETECTORS)
    return parser.parse_args()


def main():
    args = parse_args()
    detectors = tuple(args.detector) if args.detector else DETECTORS
    loaded = {}
    for detector in detectors:
        emcee_root = args.statistics_dir / "production" / detector.lower() / "emcee"
        chain_path = emcee_root / "emcee_chain.npz"
        chain, log_probability, acceptance = load_chain(chain_path)
        tau = integrated_time(chain)
        plot_parameter_traces(emcee_root / "emcee_parameter_traces.png", detector, chain)
        plot_objective_trace(emcee_root / "emcee_objective_trace.png", detector, log_probability)
        write_summary(
            emcee_root / "emcee_convergence_summary.md",
            detector,
            chain,
            log_probability,
            acceptance,
            tau,
        )
        loaded[detector] = (chain, log_probability, acceptance, tau)
        print(f"Wrote EMCEE diagnostics for {detector}", flush=True)
    if len(loaded) > 1:
        comparison = args.statistics_dir / "production" / "wfi_production_emcee_objective_comparison.png"
        plot_objective_comparison(comparison, loaded)
        print(f"Wrote {comparison}")


if __name__ == "__main__":
    main()
