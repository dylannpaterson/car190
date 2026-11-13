"""Plot the fixed and jointly calibrated all-LED fit results."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
HARMONIC_DIR = ROOT / "wfi01_joint_optical_soc_core_phase_final"
Q_DIR = ROOT / "wfi01_joint_complex_q_soc_core_final"
FIXED_NODES = ROOT / "wfi01_all_pixel_acm_step_baseline" / "wfi01_all_pixel_soc_global_nodes.csv"
Q_TABLE = ROOT / "effective_sca1_reflectance_physical" / "effective_sca1_reflectance_table.csv"
DEFAULT_OUTPUT = ROOT / "joint_fit_plots"


def read_predictions(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        led = int(row["led"])
        grouped.setdefault(led, []).append(
            {
                "days": float(row["days"]),
                "observed": float(row["observed_log_change"]),
                "predicted": float(row["predicted_log_change"]),
                "residual": float(row["residual_log_change"]),
            }
        )
    for values in grouped.values():
        values.sort(key=lambda row: row["days"])
    return grouped


def read_q_table(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    wavelength = np.asarray([float(row["wavelength_nm"]) for row in rows])
    rho = np.asarray([float(row["rho"]) for row in rows])
    phase = np.asarray([float(row["phase_offset_rad"]) for row in rows])
    alpha = np.asarray([float(row["alpha_per_nm"]) for row in rows])
    return wavelength, rho * np.exp(1j * phase), alpha


def read_fixed_nodes(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "days": float(row["days"]),
            "accumulation": float(
                row["incremental_thickness_nm"]
                if "incremental_thickness_nm" in row
                else row["fitted_thickness_nm"]
            ),
        }
        for row in rows
        if row["sed_mode"] == "core" and row["selected"].lower() == "true"
    ]


def read_q_nodes(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    nodes = []
    for row in rows:
        nodes.append(
            {
                "days": float(row["days"]),
                "accumulation": float(
                    row["incremental_thickness_nm"]
                    if "incremental_thickness_nm" in row
                    else row["thickness_nm"]
                ),
            }
        )
    return nodes


def reference_anchored_series(nodes):
    days = np.asarray([row["days"] for row in nodes], dtype=float)
    accumulation = np.asarray(
        [row["accumulation"] for row in nodes], dtype=float
    )
    if days.size and days[0] > 0.0:
        days = np.concatenate(([0.0], days))
        accumulation = np.concatenate(([0.0], accumulation))
    return days, accumulation


def parse_summary_parameters(path):
    text = path.read_text()

    def value(label):
        match = re.search(rf"\| {re.escape(label)} \| ([^|]+) \|", text)
        if match is None:
            raise ValueError(f"Missing {label} in {path}")
        return float(match.group(1).strip())

    return {
        "q_log_rho_scale": value("log q magnitude scale"),
        "q_rho_slope": value("q magnitude slope"),
        "q_phase_offset": value("q phase offset (rad)"),
        "q_phase_slope": value("q phase slope"),
        "q_alpha_scale": value("alpha scale"),
        "q_alpha_add": value("added alpha per 50 nm"),
    }


def percent(value):
    return 100.0 * np.expm1(value)


def plot_fit_comparison(output, fixed, harmonic, q):
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True, constrained_layout=True)
    for led, axis in enumerate(axes.flat, start=1):
        observed = fixed[led]
        days = np.asarray([row["days"] for row in observed])
        axis.plot(days, percent([row["observed"] for row in observed]), "ko", ms=3.5, label="Observed")
        axis.plot(days, percent([row["predicted"] for row in fixed[led]]), color="#777777", lw=1.5, label="Fixed empirical")
        axis.plot(days, percent([row["predicted"] for row in harmonic[led]]), color="#0072B2", lw=1.5, label="Joint periodic")
        axis.plot(days, percent([row["predicted"] for row in q[led]]), color="#D55E00", lw=1.5, label="Joint complex q")
        axis.axhline(0.0, color="black", lw=0.6)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Change (%)")
    axes[1, 0].set_ylabel("Change (%)")
    axes[1, 1].set_xlabel("Days since first exposure")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("WFI01 all-LED fits with one shared ice trajectory and ACM step")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_residuals(output, fixed, harmonic, q):
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True, constrained_layout=True)
    for led, axis in enumerate(axes.flat, start=1):
        for values, color, label in (
            (fixed[led], "#777777", "Fixed empirical"),
            (harmonic[led], "#0072B2", "Joint periodic"),
            (q[led], "#D55E00", "Joint complex q"),
        ):
            axis.plot(
                [row["days"] for row in values],
                percent([row["residual"] for row in values]),
                "o-",
                ms=2.5,
                lw=1.0,
                color=color,
                label=label,
            )
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_title(f"LED {led}")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Observed - model (%)")
    axes[1, 0].set_ylabel("Observed - model (%)")
    axes[1, 1].set_xlabel("Days since first exposure")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Per-exposure residuals")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_q_parameters(output, q_table, parameters):
    wavelength, q_base, alpha_base = q_table
    x = (wavelength - 1400.0) / 1000.0
    q_fit = q_base * np.exp(
        parameters["q_log_rho_scale"]
        + parameters["q_rho_slope"] * x
        + 1j * (parameters["q_phase_offset"] + parameters["q_phase_slope"] * x)
    )
    alpha_fit = parameters["q_alpha_scale"] * alpha_base + parameters["q_alpha_add"] / 50.0
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(wavelength, np.abs(q_base), color="#777777", label="Table-inferred q")
    axes[0].plot(wavelength, np.abs(q_fit), color="#D55E00", label="Joint fitted q")
    axes[0].set_ylabel("|q|")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(wavelength, np.unwrap(np.angle(q_base)), color="#777777")
    axes[1].plot(wavelength, np.unwrap(np.angle(q_fit)), color="#D55E00")
    axes[1].set_ylabel("arg(q) (rad, unwrapped)")
    axes[1].grid(alpha=0.2)
    axes[2].plot(wavelength, alpha_base * wavelength / (4.0 * np.pi), color="#777777", label="Table k_eff")
    axes[2].plot(wavelength, alpha_fit * wavelength / (4.0 * np.pi), color="#D55E00", label="Joint k_eff")
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].set_ylabel("k_eff")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.2)
    figure.suptitle("Effective complex-q calibration")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_ice_trajectories(output, fixed_nodes, q_nodes):
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fixed_days, fixed_accumulation = reference_anchored_series(fixed_nodes)
    axis.plot(
        fixed_days,
        fixed_accumulation,
        "o-",
        color="#777777",
        label="Fixed empirical response; accumulation",
    )
    q_days, q_accumulation = reference_anchored_series(q_nodes)
    axis.plot(
        q_days,
        q_accumulation,
        "o-",
        color="#D55E00",
        label="Joint complex-q response; accumulation",
    )
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("Accumulated thickness Delta d (nm)")
    axis.set_title("Accumulation relative to the first/reference epoch")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def print_metrics(name, rows):
    print(name)
    for led in range(1, 7):
        residual = np.asarray([row["residual"] for row in rows[led]])
        print(
            f"  LED {led}: RMS={np.sqrt(np.mean(percent(residual) ** 2)):.6g}% "
            f"final={percent(residual[-1]):+.6g}%"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--harmonic-dir", type=Path, default=HARMONIC_DIR)
    parser.add_argument("--q-dir", type=Path, default=Q_DIR)
    parser.add_argument("--q-table", type=Path, default=Q_TABLE)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = read_predictions(args.harmonic_dir / "baseline_predictions.csv")
    harmonic = read_predictions(args.harmonic_dir / "joint_np_predictions.csv")
    q = read_predictions(args.q_dir / "joint_q_np_predictions.csv")
    parameters = parse_summary_parameters(args.q_dir / "joint_complex_q_fit.md")
    plot_fit_comparison(args.output_dir / "all_led_fit_comparison.png", fixed, harmonic, q)
    plot_residuals(args.output_dir / "all_led_residuals.png", fixed, harmonic, q)
    plot_q_parameters(args.output_dir / "complex_q_parameters.png", read_q_table(args.q_table), parameters)
    plot_ice_trajectories(
        args.output_dir / "ice_trajectory_comparison.png",
        read_fixed_nodes(FIXED_NODES),
        read_q_nodes(args.q_dir / "joint_q_nodes.csv"),
    )
    print_metrics("Fixed empirical", fixed)
    print_metrics("Joint periodic", harmonic)
    print_metrics("Joint complex q", q)
    print(f"Wrote {args.output_dir / 'all_led_fit_comparison.png'}")
    print(f"Wrote {args.output_dir / 'all_led_residuals.png'}")
    print(f"Wrote {args.output_dir / 'complex_q_parameters.png'}")
    print(f"Wrote {args.output_dir / 'ice_trajectory_comparison.png'}")


if __name__ == "__main__":
    main()
