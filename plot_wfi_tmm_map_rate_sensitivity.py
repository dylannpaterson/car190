"""Plot WFI01/WFI10 MAP accumulation-rate sensitivity on one time axis."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_PRODUCTION_ROOT = (
    ROOT
    / "20261012/corrected_statistics_program1063_4h/production/tile_size_256"
)
DETECTORS = ("wfi01", "wfi10")
RUNS = (
    ("n_prior_sigma_0p25", "n sigma = 0.25", "#4C78A8"),
    ("n_prior_sigma_0p075", "n sigma = 0.075", "#F58518"),
    ("n_prior_sigma_0p05", "n sigma = 0.05", "#E45756"),
    ("n_fixed_zero", "n log scale = 0", "#54A24B"),
)


def load_rate_series(production_root, detector, run_name, tile_id):
    path = (
        Path(production_root)
        / detector
        / run_name
        / "fits"
        / f"tile_{tile_id:03d}"
        / "posterior_samples.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as samples:
        days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
        rates = np.asarray(
            samples["tile_instantaneous_rate_nm_day"], dtype=float
        ).reshape(-1)
        reset_days = np.asarray(samples["ice_reset_day"], dtype=float).reshape(-1)
    if days.shape != rates.shape or days.size == 0:
        raise ValueError(f"Unexpected rate series shape in {path}")
    if reset_days.size != 1 or not np.isfinite(reset_days[0]):
        raise ValueError(f"Unexpected reset day in {path}")
    order = np.argsort(days, kind="stable")
    return path, days[order], rates[order], float(reset_days[0])


def write_rate_table(path, series):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("detector", "model", "days", "rate_nm_day"),
        )
        writer.writeheader()
        for detector, _run_name, model, days, rates in series:
            for day, rate in zip(days, rates):
                writer.writerow(
                    {
                        "detector": detector.upper(),
                        "model": model,
                        "days": float(day),
                        "rate_nm_day": float(rate),
                    }
                )


def make_plot(path, series, tile_id, reset_day):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)
    for detector, _run_name, model, days, rates in series:
        color = next(color for _name, label, color in RUNS if label == model)
        axis.plot(
            days,
            rates,
            color=color,
            linestyle="-" if detector == "wfi01" else "--",
            linewidth=1.8,
            alpha=0.95,
            label=f"{detector.upper()} | {model}",
        )

    axis.axvline(
        reset_day,
        color="0.35",
        linestyle=":",
        linewidth=1.2,
        label="Program 1063 Decon2",
    )
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("Instantaneous accumulation rate (nm/day)")
    axis.set_title(f"Tile {tile_id} rate sensitivity")
    axis.grid(color="0.88", linewidth=0.7)
    axis.set_axisbelow(True)

    detector_handles = [
        Line2D([0], [0], color="0.15", linestyle="-", linewidth=1.8, label="WFI01"),
        Line2D([0], [0], color="0.15", linestyle="--", linewidth=1.8, label="WFI10"),
    ]
    prior_handles = [
        Line2D([0], [0], color=color, linewidth=2.2, label=label)
        for _name, label, color in RUNS
    ]
    reset_handle = Line2D(
        [0],
        [0],
        color="0.35",
        linestyle=":",
        linewidth=1.2,
        label="Program 1063 Decon2",
    )
    first_legend = axis.legend(
        handles=detector_handles,
        title="Detector",
        loc="upper left",
        frameon=False,
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=prior_handles + [reset_handle],
        title="HgCdTe n treatment",
        loc="upper right",
        frameon=False,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-root",
        type=Path,
        default=DEFAULT_PRODUCTION_ROOT,
        help="Production tile-size root containing the detector sensitivity runs.",
    )
    parser.add_argument("--tile-id", type=int, default=136)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PRODUCTION_ROOT / "wfi01_wfi10_tile136_rate_sensitivity.png",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_PRODUCTION_ROOT / "wfi01_wfi10_tile136_rate_sensitivity.csv",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    series = []
    reset_days = []
    for detector in DETECTORS:
        for run_name, model, _color in RUNS:
            _path, days, rates, reset_day = load_rate_series(
                args.production_root,
                detector,
                run_name,
                args.tile_id,
            )
            series.append((detector, run_name, model, days, rates))
            reset_days.append(reset_day)
    if not np.allclose(reset_days, reset_days[0]):
        raise ValueError(f"MAP archives disagree on reset day: {reset_days}")
    write_rate_table(args.csv_output, series)
    make_plot(args.output, series, args.tile_id, reset_days[0])
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
