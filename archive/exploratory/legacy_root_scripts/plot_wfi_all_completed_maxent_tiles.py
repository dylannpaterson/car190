"""Plot all completed MaxEnt WFI tile trajectories in one figure."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_STATISTICS_DIR = ROOT / "20261008/corrected_statistics_program1063_4h"
DEFAULT_RESET_DAY = 12.907397928240743
DETECTORS = tuple(f"WFI{index:02d}" for index in range(1, 19))


def _is_maxent_archive(samples):
    try:
        return (
            str(np.asarray(samples["trajectory_regularization"]).item()) == "maxent"
            and np.isclose(
                float(np.asarray(samples["trajectory_entropy_weight"]).item()),
                0.1,
                rtol=0.0,
                atol=1.0e-12,
            )
            and int(np.asarray(samples["constant_rate_map_warmup"]).item()) == 1
        )
    except (KeyError, OSError, ValueError):
        return False


def _node_days(samples):
    days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
    repeats = np.asarray(samples["observation_repeat"], dtype=int).reshape(-1)
    return np.asarray(
        [np.mean(days[repeats == repeat]) for repeat in sorted(set(repeats.tolist()))],
        dtype=float,
    )


def _model_run_date(path):
    """Use the UTC archive date when the fit does not store a run timestamp."""
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat()


def load_completed_tiles(statistics_dir):
    production_root = Path(statistics_dir) / "production"
    records = []
    for detector in DETECTORS:
        fit_root = production_root / detector.lower() / "fits"
        for posterior_path in sorted(fit_root.glob("tile_*/posterior_samples.npz")):
            with np.load(posterior_path, allow_pickle=False) as samples:
                if not _is_maxent_archive(samples):
                    continue
                tile_ids = np.asarray(samples["tile_ids"], dtype=int).reshape(-1)
                node_days = _node_days(samples)
                rates = np.asarray(
                    samples["node_instantaneous_rate_nm_day"], dtype=float
                ).reshape(-1)
                thickness = np.asarray(samples["np_node_thickness"], dtype=float).reshape(-1)
                if tile_ids.size != 1 or rates.size != node_days.size or thickness.size != node_days.size:
                    raise ValueError(f"Unexpected trajectory shape in {posterior_path}")
                records.append(
                    {
                        "detector": detector,
                        "tile_id": int(tile_ids[0]),
                        "model_run_date": _model_run_date(posterior_path),
                        "node_days": node_days,
                        "rates": rates,
                        "thickness": thickness,
                    }
                )
    records.sort(key=lambda record: (record["detector"], record["tile_id"]))
    if not records:
        raise ValueError(f"No completed MaxEnt archives found below {production_root}")
    reference_days = records[0]["node_days"]
    for record in records[1:]:
        if not np.allclose(record["node_days"], reference_days, rtol=0.0, atol=1.0e-12):
            raise ValueError("Completed MaxEnt archives use different node-day grids")
    return records


def plot_records(records, output, reset_day):
    count = len(records)
    run_dates = sorted({record["model_run_date"] for record in records})
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    colors = plt.get_cmap("turbo")
    normalizer = Normalize(vmin=0, vmax=max(count - 1, 1))
    detector_centers = []
    detector_labels = []
    start = 0
    for detector in DETECTORS:
        detector_records = [record for record in records if record["detector"] == detector]
        if not detector_records:
            continue
        detector_centers.append(start + 0.5 * (len(detector_records) - 1))
        detector_labels.append(detector)
        start += len(detector_records)

    for index, record in enumerate(records):
        color = colors(normalizer(index))
        label = f"{record['detector']} tile {record['tile_id']:03d}"
        axes[0].plot(
            record["node_days"],
            record["rates"],
            color=color,
            alpha=0.42,
            linewidth=0.65,
            label=label,
        )
        axes[1].plot(
            record["node_days"],
            record["thickness"],
            color=color,
            alpha=0.42,
            linewidth=0.65,
        )

    for axis in axes:
        axis.axvline(
            reset_day,
            color="0.35",
            linestyle=":",
            linewidth=0.9,
            alpha=0.8,
        )
        axis.set_xlabel("Days since first exposure")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Instantaneous rate (nm/day)")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_title("All completed tile rates")
    axes[1].set_ylabel("Ice thickness (nm)")
    axes[1].set_title("All completed tile thicknesses")
    figure.suptitle(
        f"All completed MaxEnt WFI tile trajectories ({count} tiles)\n"
        f"Model run date(s): {', '.join(run_dates)}"
    )
    scalar_mappable = ScalarMappable(norm=normalizer, cmap=colors)
    scalar_mappable.set_array([])
    colorbar = figure.colorbar(scalar_mappable, ax=axes, pad=0.02)
    colorbar.set_label("Detector/tile sequence")
    colorbar.set_ticks(detector_centers)
    colorbar.set_ticklabels(detector_labels)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_detector_rate_envelopes(records, output, reset_day):
    """Plot each detector's median rate with its 16th/84th percentile envelope."""
    detectors = sorted({record["detector"] for record in records})
    run_dates = sorted({record["model_run_date"] for record in records})
    node_days = records[0]["node_days"]
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    colors = plt.get_cmap("tab20")

    for index, detector in enumerate(detectors):
        detector_records = [
            record for record in records if record["detector"] == detector
        ]
        rates = np.asarray([record["rates"] for record in detector_records])
        lower, median, upper = np.percentile(rates, [16.0, 50.0, 84.0], axis=0)
        color = colors(index / max(len(detectors) - 1, 1))
        label = f"{detector} (n={len(detector_records)})"
        axis.fill_between(node_days, lower, upper, color=color, alpha=0.16)
        axis.plot(node_days, median, color=color, linewidth=1.8, label=label)

    axis.axvline(
        reset_day,
        color="0.35",
        linestyle=":",
        linewidth=0.9,
        alpha=0.8,
        label="Decon2",
    )
    axis.set_xlabel("Days since first exposure")
    axis.set_ylabel("Instantaneous ice accumulation rate (nm/day)")
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.2)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.suptitle(
        f"MaxEnt WFI detector rate envelopes ({len(records)} tiles)\n"
        f"Median with 16th/84th percentile envelope; "
        f"model run date(s): {', '.join(run_dates)}"
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, default=DEFAULT_STATISTICS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reset-day", type=float, default=DEFAULT_RESET_DAY)
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_completed_tiles(args.statistics_dir)
    output = args.output or (
        Path(args.statistics_dir) / "production" / "wfi_all_completed_maxent_tile_trajectories.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_records(records, output, args.reset_day)
    envelope_output = output.with_name("wfi_all_completed_maxent_detector_rate_envelopes.png")
    plot_detector_rate_envelopes(records, envelope_output, args.reset_day)
    print(f"Wrote {output}")
    print(f"Wrote {envelope_output}")
    print(f"Plotted {len(records)} completed MaxEnt tiles")


if __name__ == "__main__":
    main()
