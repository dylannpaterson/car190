"""Plot ice trajectories and instantaneous rates from a regularization experiment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = (
    ROOT
    / "20261008/corrected_statistics_program1063_4h/production/wfi02/"
    "regularization_experiment/tile_001"
)
VARIANTS = ("baseline", "curvature_off", "maxent")
LABELS = {
    "baseline": "Baseline curvature",
    "curvature_off": "Curvature off",
    "maxent": "MaxEnt prior",
}
COLORS = {
    "baseline": "tab:blue",
    "curvature_off": "tab:orange",
    "maxent": "tab:green",
}
LINESTYLES = {
    "baseline": "-",
    "curvature_off": "--",
    "maxent": "-.",
}


def _sample_series(archive, name, count):
    values = np.asarray(archive[name], dtype=float)
    if values.size % count:
        raise ValueError(f"{name} does not contain complete observation series")
    return values.reshape(-1, count)[0]


def load_variant(output_dir):
    output_dir = Path(output_dir)
    with np.load(output_dir / "posterior_samples.npz", allow_pickle=False) as archive:
        days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
        thickness = _sample_series(archive, "tile_ice_thickness", days.size)
        rate = _sample_series(
            archive,
            "tile_instantaneous_rate_nm_day",
            days.size,
        )
        reset_day = float(np.asarray(archive["ice_reset_day"]).item())
    if np.any(np.diff(days) <= 0.0):
        raise ValueError("Observation days must be strictly increasing")
    if not np.all(np.isfinite(thickness)) or not np.all(np.isfinite(rate)):
        raise ValueError(f"Non-finite trajectory values in {output_dir}")
    return days, thickness, rate, reset_day


def _plot_series(
    experiment_root,
    variants,
    output_path,
    series_index,
    ylabel,
    title,
):
    loaded = {
        variant: load_variant(Path(experiment_root) / variant)
        for variant in variants
    }
    reset_days = {round(values[3], 12) for values in loaded.values()}
    if len(reset_days) != 1:
        raise ValueError("Experiment variants have inconsistent reset days")
    reset_day = next(iter(reset_days))
    metadata_path = Path(experiment_root) / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}

    figure, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for variant in variants:
        days, thickness, rate, _ = loaded[variant]
        values = (thickness, rate)[series_index]
        axis.plot(
            days,
            values,
            color=COLORS[variant],
            linestyle=LINESTYLES[variant],
            linewidth=2.0,
            label=LABELS[variant],
        )
    axis.axvline(
        reset_day,
        color="0.25",
        linestyle=":",
        linewidth=1.1,
        label=f"decon2 day {reset_day:.3f}",
    )
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axis.legend(frameon=False)
    source_tile = metadata.get("source_tile")
    target_tile = metadata.get("target_tile")
    if source_tile is not None and target_tile is not None:
        axis.text(
            0.01,
            0.02,
            f"target tile {int(target_tile):03d}; EMCEE source tile {int(source_tile):03d}",
            transform=axis.transAxes,
            fontsize=9,
            color="0.35",
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    return parser.parse_args()


def main():
    args = parse_args()
    variants = tuple(args.variant) if args.variant else VARIANTS
    output_dir = args.output_dir or args.experiment_root
    _plot_series(
        args.experiment_root,
        variants,
        output_dir / "ice_trajectory_comparison.png",
        0,
        "Ice thickness [nm]",
        "WFI02 tile 001 ice trajectory by regularization",
    )
    _plot_series(
        args.experiment_root,
        variants,
        output_dir / "instantaneous_ice_accumulation_rate_comparison.png",
        1,
        "Instantaneous ice accumulation rate [nm/day]",
        "WFI02 tile 001 instantaneous ice accumulation rate",
    )
    print(f"Wrote {output_dir / 'ice_trajectory_comparison.png'}")
    print(
        "Wrote "
        f"{output_dir / 'instantaneous_ice_accumulation_rate_comparison.png'}"
    )


if __name__ == "__main__":
    main()
