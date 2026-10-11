"""Check constant-rate MAP dependence on the starting rate for tile 000."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fit_wfi_tmm_production as production
from run_wfi_constant_rate_ice_modulation import (
    load_constant_rate_initial_values,
)


DETECTORS = ("wfi01", "wfi02")
DEFAULT_START_RATES = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--constant-rate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--start-rates",
        type=float,
        nargs="+",
        default=DEFAULT_START_RATES,
    )
    return parser.parse_args()


def fit_detector(detector, args, start_rates):
    statistics_path = args.statistics_dir / detector / "tile_statistics" / (
        f"{detector}_tile_log_statistics_512_tile_000.csv"
    )
    map_path = (
        args.constant_rate_dir / detector / "fits" / "tile_000" / "constant_rate_map.npz"
    )
    data = production.load_tile_statistics(statistics_path)
    base_values = load_constant_rate_initial_values(
        map_path,
        data,
        production.ICE_N_MODULATION_PRIOR_SIGMA_LOG,
    )
    base_values.pop("ice_n_log_modulation")
    rows = []
    for start_rate in start_rates:
        initial_values = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in base_values.items()
        }
        initial_values["log_rate_scale"] = np.full_like(
            initial_values["log_rate_scale"],
            np.log(start_rate),
        )
        config = production.ProductionConfig(
            tile_statistics=statistics_path,
            output_dir=args.output_dir / detector / f"start_{start_rate:g}",
            map_initial_values=initial_values,
            constant_rate=True,
            model_metadata={
                "constant_rate_multistart_diagnostic": 1,
                "initial_rate_nm_day": start_rate,
            },
        )
        _, _, map_values, diagnostics, _ = production.fit_map_state(config)
        final_rate = float(
            np.exp(np.asarray(map_values["log_rate_scale"]).reshape(-1)[0])
        )
        rows.append(
            {
                "detector": detector,
                "initial_rate_nm_day": float(start_rate),
                "final_rate_nm_day": final_rate,
                "objective": float(diagnostics["objective"]),
                "least_squares_cost": float(diagnostics["objective_history"][-1]),
                "iterations": int(diagnostics["iterations"]),
                "gradient_converged": bool(diagnostics["gradient_converged"]),
                "scaled_gradient_norm": float(diagnostics["scaled_gradient_norm"]),
            }
        )
        print(
            f"{detector} start={start_rate:g} -> rate={final_rate:.9g}, "
            f"cost={rows[-1]['least_squares_cost']:.9g}",
            flush=True,
        )
    return rows


def write_outputs(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "multistart_summary.csv").open("w", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "multistart_summary.json").write_text(json.dumps(rows, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for detector, color in (("wfi01", "#0072b2"), ("wfi02", "#d55e00")):
        selected = [row for row in rows if row["detector"] == detector]
        selected.sort(key=lambda row: row["initial_rate_nm_day"])
        axes[0].plot(
            [row["initial_rate_nm_day"] for row in selected],
            [row["final_rate_nm_day"] for row in selected],
            marker="o",
            color=color,
            label=detector.upper(),
        )
        axes[1].plot(
            [row["initial_rate_nm_day"] for row in selected],
            [row["least_squares_cost"] for row in selected],
            marker="o",
            color=color,
            label=detector.upper(),
        )
    limits = axes[0].get_xlim()
    axes[0].plot(limits, limits, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Constant-rate multistart")
    axes[0].set_xlabel("initial rate [nm/day]")
    axes[0].set_ylabel("final rate [nm/day]")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].set_title("Final least-squares cost")
    axes[1].set_xlabel("initial rate [nm/day]")
    axes[1].set_ylabel("0.5 sum(residuals^2)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    figure.savefig(output_dir / "constant_rate_multistart.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if any(not np.isfinite(rate) or rate <= 0.0 for rate in args.start_rates):
        raise ValueError("--start-rates must contain positive finite values")
    jax, _, _, _ = production.require_numpyro()
    jax.config.update("jax_enable_x64", True)
    rows = []
    for detector in DETECTORS:
        rows.extend(fit_detector(detector, args, args.start_rates))
    write_outputs(rows, args.output_dir)
    print(f"Wrote multistart results to {args.output_dir}")


if __name__ == "__main__":
    main()
