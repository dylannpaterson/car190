"""Plot tile-mean ice/rate trajectories for the 18 production WFI fits."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
SCAS = tuple(f"WFI{index:02d}" for index in range(1, 19))
DEFAULT_FIT_SUFFIX = (
    "production_sharedAR_tileRate_twoCommonACM_"
    "map2000_warm100_sample500_depth4_5nm"
)
DEFAULT_RESET_DAY = 15.064405023148149


def _one_tile_series(samples, key):
    values = np.asarray(samples[key], dtype=float)
    if values.ndim == 3:
        values = values[0, 0]
    elif values.ndim == 2:
        values = values[0]
    return values.reshape(-1)


def load_independent_trajectory(output_dir):
    fit_dirs = sorted(
        (Path(output_dir) / "fits").glob("tile_*_least_squares")
    )
    if not fit_dirs:
        raise FileNotFoundError(f"No independent tile fits found in {output_dir}")
    thickness = []
    rates = []
    days = None
    for fit_dir in fit_dirs:
        with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
            current_days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
            if days is None:
                days = current_days
            elif not np.array_equal(days, current_days):
                raise ValueError(f"Observation days differ in {fit_dir}")
            thickness.append(_one_tile_series(samples, "tile_ice_thickness"))
            rates.append(_one_tile_series(samples, "tile_instantaneous_rate_nm_day"))
    thickness = np.asarray(thickness, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if thickness.shape[1] != days.size or rates.shape != thickness.shape:
        raise ValueError(f"Unexpected independent trajectory shape in {output_dir}")
    order = np.argsort(days, kind="stable")
    return (
        days[order],
        np.percentile(thickness[:, order], [16, 50, 84], axis=0),
        np.percentile(rates[:, order], [16, 50, 84], axis=0),
    )


def load_trajectory(output_dir):
    output_dir = Path(output_dir)
    if not (output_dir / "posterior_samples.npz").is_file():
        return load_independent_trajectory(output_dir)
    with np.load(output_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        days = np.asarray(samples["observation_days"], dtype=float)
        ice = np.asarray(samples["ice_thickness"], dtype=float)
        if ice.ndim == 3:
            ice = np.mean(ice, axis=1)
        if ice.ndim != 2 or ice.shape[1] != len(days):
            raise ValueError(f"Unexpected ice-thickness shape in {output_dir}")

        rates = np.asarray(samples["instantaneous_rate_nm_day"], dtype=float)
        if rates.ndim == 3:
            rates = np.mean(rates, axis=1)
        if rates.ndim != 2 or rates.shape != ice.shape:
            raise ValueError(f"Unexpected rate shape in {output_dir}")

    order = np.argsort(days, kind="stable")
    days = days[order]
    ice_quantiles = np.percentile(ice[:, order], [16, 50, 84], axis=0)
    rate_quantiles = np.percentile(rates[:, order], [16, 50, 84], axis=0)
    return days, ice_quantiles, rate_quantiles


def write_quantiles(path, trajectories):
    fields = (
        "detector",
        "days",
        "ice_thickness_p16_nm",
        "ice_thickness_median_nm",
        "ice_thickness_p84_nm",
        "mean_tile_rate_p16_nm_day",
        "mean_tile_rate_median_nm_day",
        "mean_tile_rate_p84_nm_day",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for detector, (days, ice, rates) in trajectories.items():
            for index, day in enumerate(days):
                writer.writerow(
                    {
                        "detector": detector,
                        "days": day,
                        "ice_thickness_p16_nm": ice[0, index],
                        "ice_thickness_median_nm": ice[1, index],
                        "ice_thickness_p84_nm": ice[2, index],
                        "mean_tile_rate_p16_nm_day": rates[0, index],
                        "mean_tile_rate_median_nm_day": rates[1, index],
                        "mean_tile_rate_p84_nm_day": rates[2, index],
                    }
                )


def plot_all_detector_trajectories(
    path, trajectories, reset_day=None, independent_fit=False
):
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True)
    colors = plt.get_cmap("tab20", len(SCAS))
    legend_handles = []
    for index, detector in enumerate(SCAS):
        days, ice, rates = trajectories[detector]
        color = colors(index)
        axes[0].fill_between(days, ice[0], ice[2], color=color, alpha=0.08)
        axes[0].plot(days, ice[1], color=color, linewidth=1.4)
        axes[1].fill_between(days, rates[0], rates[2], color=color, alpha=0.08)
        axes[1].plot(days, rates[1], color=color, linewidth=1.4)
        legend_handles.append(
            Line2D([], [], color=color, linewidth=2.0, label=detector)
        )

    if reset_day is not None:
        for axis in axes:
            axis.axvline(
                reset_day,
                color="0.35",
                linestyle=":",
                linewidth=0.9,
                alpha=0.8,
                label="Ice reset",
            )

    axes[0].set_ylabel("Tile-mean ice thickness (nm)")
    axes[0].set_title("Ice trajectory")
    axes[1].set_ylabel("Mean tile accumulation rate (nm/day)")
    axes[1].set_title("Accumulation rate")
    for axis in axes:
        axis.set_xlabel("Days since first exposure")
        axis.set_xlim(left=0.0)
        axis.grid(alpha=0.2)
    title = "WFI01–WFI18 median trajectories with 16th–84th percentile bands"
    if independent_fit:
        title = "WFI01–WFI18 independent-fit trajectories; bands span tiles"
    figure.suptitle(title, fontsize=14)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.93))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fit-suffix", default=DEFAULT_FIT_SUFFIX)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--reset-day",
        type=float,
        default=None,
        help="Mark the ice reset day on both panels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trajectories = {}
    for detector in SCAS:
        output_dir = args.root / f"{detector.lower()}_{args.fit_suffix}"
        posterior_path = output_dir / "posterior_samples.npz"
        independent_paths = (output_dir / "fits").glob("tile_*_least_squares/posterior_samples.npz")
        if not posterior_path.is_file() and not any(independent_paths):
            raise FileNotFoundError(f"Missing production posterior: {posterior_path}")
        trajectories[detector] = load_trajectory(output_dir)
    output = args.output or args.root / "wfi_all_detector_ice_rate_trajectories.png"
    quantile_path = output.with_name(output.stem + "_quantiles.csv")
    independent_fit = "independent_tiles" in args.fit_suffix
    plot_all_detector_trajectories(
        output,
        trajectories,
        reset_day=args.reset_day,
        independent_fit=independent_fit,
    )
    write_quantiles(quantile_path, trajectories)
    print(f"Wrote {output}")
    print(f"Wrote {quantile_path}")


if __name__ == "__main__":
    main()
