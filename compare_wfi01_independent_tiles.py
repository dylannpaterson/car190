"""Compare independent WFI01 tile fits with the hierarchical all-tile fit."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_COMBINED = (
    ROOT
    / "wfi01_absolute_substrate_power_hgcdte_prior025_all_tiles_map10000_emcee1000_warmstarted"
)
DEFAULT_INDEPENDENT_ROOT = (
    ROOT / "wfi01_absolute_substrate_power_hgcdte_prior025_independent_tiles"
)
DEFAULT_TILE_ZERO = (
    ROOT
    / "wfi01_absolute_substrate_power_hgcdte_prior025_one_tile_map10000_emcee1000_warmstarted"
)


def posterior_median(archive, key):
    return np.median(np.asarray(archive[key], dtype=float), axis=0)


def select_tile(value, tile_index, tile_count):
    value = np.asarray(value)
    if value.ndim == 0:
        return value.item()
    if value.ndim >= 2 and value.shape[0] == 1 and value.shape[1] == tile_count:
        return value[0, tile_index]
    if value.shape[0] == tile_count:
        return value[tile_index]
    if tile_count == 1 and value.shape[0] == 1:
        return value[0]
    raise ValueError(
        f"Cannot select tile {tile_index} from archived shape {value.shape}"
    )


def prediction_rms(path, tile_id):
    residuals = []
    with (Path(path) / "tile_group_predictions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["tile_id"]) == tile_id:
                residuals.append(float(row["residual_log"]))
    if not residuals:
        raise ValueError(f"No predictions found for tile {tile_id} in {path}")
    residuals = np.asarray(residuals, dtype=float)
    return float(np.sqrt(np.mean(residuals * residuals)))


def archive_values(path, tile_id):
    with np.load(Path(path) / "posterior_samples.npz", allow_pickle=False) as archive:
        tile_ids = np.asarray(archive["tile_ids"], dtype=int).reshape(-1)
        tile_index = int(np.flatnonzero(tile_ids == tile_id)[0])
        tile_count = len(tile_ids)
        ar = select_tile(posterior_median(archive, "ar_parameters"), tile_index, tile_count)
        initial_thickness = select_tile(
            posterior_median(archive, "initial_thickness_nm"), tile_index, tile_count
        )
        reset_thickness = select_tile(
            posterior_median(archive, "reset_initial_thickness_nm"),
            tile_index,
            tile_count,
        )
        initial_rate = select_tile(
            posterior_median(archive, "tile_initial_rate_nm_day"), tile_index, tile_count
        )
        final_rate = select_tile(
            posterior_median(archive, "tile_final_rate_nm_day"), tile_index, tile_count
        )
        mean_rate = select_tile(
            posterior_median(archive, "tile_mean_rate_nm_day"), tile_index, tile_count
        )
        ice_thickness = select_tile(
            posterior_median(archive, "tile_ice_thickness"), tile_index, tile_count
        )
        instantaneous_rate = select_tile(
            posterior_median(archive, "tile_instantaneous_rate_nm_day"),
            tile_index,
            tile_count,
        )
        return {
            "tile_row": int(np.asarray(archive["tile_ids"]).reshape(-1)[tile_index] // 8),
            "tile_col": int(np.asarray(archive["tile_ids"]).reshape(-1)[tile_index] % 8),
            "ar": np.asarray(ar, dtype=float).reshape(-1),
            "initial_thickness": float(initial_thickness),
            "reset_thickness": float(reset_thickness),
            "initial_rate": float(initial_rate),
            "final_rate": float(final_rate),
            "mean_rate": float(mean_rate),
            "hgcdte_n_log": float(np.median(np.asarray(archive["absolute_hgcdte_n_log_scale"]))),
            "hgcdte_k_log": float(np.median(np.asarray(archive["absolute_hgcdte_k_log_scale"]))),
            "incidence_angle": float(np.median(np.asarray(archive["incidence_angle_deg"]))),
            "days": np.asarray(archive["observation_days"], dtype=float).reshape(-1),
            "led": np.asarray(archive["observation_led"], dtype=int).reshape(-1),
            "repeat": np.asarray(archive["observation_repeat"], dtype=int).reshape(-1),
            "ice_thickness": np.asarray(ice_thickness, dtype=float).reshape(-1),
            "instantaneous_rate": np.asarray(instantaneous_rate, dtype=float).reshape(-1),
        }


def tile_fit_paths(independent_root, tile_zero):
    paths = {0: Path(tile_zero)}
    fit_root = Path(independent_root) / "fits"
    for tile_id in range(1, 64):
        least_squares_path = fit_root / f"tile_{tile_id:03d}_least_squares"
        legacy_path = fit_root / f"tile_{tile_id:03d}_map10000_emcee1000_warmstarted"
        paths[tile_id] = (
            least_squares_path
            if (least_squares_path / "posterior_samples.npz").is_file()
            else legacy_path
        )
    return paths


def parameter_record(tile_id, independent, hierarchical, independent_path, combined_path):
    record = {
        "tile_id": tile_id,
        "tile_row": independent["tile_row"],
        "tile_col": independent["tile_col"],
        "independent_group_rms_log": prediction_rms(independent_path, tile_id),
        "hierarchical_group_rms_log": prediction_rms(combined_path, tile_id),
        "independent_initial_thickness_nm": independent["initial_thickness"],
        "hierarchical_initial_thickness_nm": hierarchical["initial_thickness"],
        "independent_reset_thickness_nm": independent["reset_thickness"],
        "hierarchical_reset_thickness_nm": hierarchical["reset_thickness"],
        "independent_initial_rate_nm_day": independent["initial_rate"],
        "hierarchical_initial_rate_nm_day": hierarchical["initial_rate"],
        "independent_final_rate_nm_day": independent["final_rate"],
        "hierarchical_final_rate_nm_day": hierarchical["final_rate"],
        "independent_mean_rate_nm_day": independent["mean_rate"],
        "hierarchical_mean_rate_nm_day": hierarchical["mean_rate"],
        "independent_hgcdte_n_log_scale": independent["hgcdte_n_log"],
        "hierarchical_hgcdte_n_log_scale": hierarchical["hgcdte_n_log"],
        "independent_hgcdte_k_log_scale": independent["hgcdte_k_log"],
        "hierarchical_hgcdte_k_log_scale": hierarchical["hgcdte_k_log"],
        "independent_incidence_angle_deg": independent["incidence_angle"],
        "hierarchical_incidence_angle_deg": hierarchical["incidence_angle"],
    }
    for layer in range(3):
        offset = 3 * layer
        for component, component_offset in (
            ("thickness", 0),
            ("n", 1),
            ("k", 2),
        ):
            record[f"independent_ar_layer{layer + 1}_{component}"] = independent["ar"][
                offset + component_offset
            ]
            record[f"hierarchical_ar_layer{layer + 1}_{component}"] = hierarchical["ar"][
                offset + component_offset
            ]
    return record


def write_parameter_table(path, records):
    fieldnames = list(records[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_trajectory_table(path, trajectories):
    fields = [
        "tile_id",
        "group",
        "days",
        "led",
        "repeat",
        "independent_ice_thickness_nm",
        "hierarchical_ice_thickness_nm",
        "ice_thickness_difference_nm",
        "independent_rate_nm_day",
        "hierarchical_rate_nm_day",
        "rate_difference_nm_day",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tile_id, independent, hierarchical in trajectories:
            for group, values in enumerate(
                zip(
                    independent["days"],
                    independent["led"],
                    independent["repeat"],
                    independent["ice_thickness"],
                    hierarchical["ice_thickness"],
                    independent["instantaneous_rate"],
                    hierarchical["instantaneous_rate"],
                )
            ):
                days, led, repeat, independent_ice, hierarchical_ice, independent_rate, hierarchical_rate = values
                writer.writerow(
                    {
                        "tile_id": tile_id,
                        "group": group,
                        "days": float(days),
                        "led": int(led),
                        "repeat": int(repeat),
                        "independent_ice_thickness_nm": float(independent_ice),
                        "hierarchical_ice_thickness_nm": float(hierarchical_ice),
                        "ice_thickness_difference_nm": float(independent_ice - hierarchical_ice),
                        "independent_rate_nm_day": float(independent_rate),
                        "hierarchical_rate_nm_day": float(hierarchical_rate),
                        "rate_difference_nm_day": float(independent_rate - hierarchical_rate),
                    }
                )


def plot_trajectory_comparison(path, trajectories):
    independent_color = "#2563eb"
    hierarchical_color = "#d97706"
    days = np.asarray(trajectories[0][1]["days"], dtype=float)
    independent_thickness = np.asarray(
        [item[1]["ice_thickness"] for item in trajectories], dtype=float
    )
    hierarchical_thickness = np.asarray(
        [item[2]["ice_thickness"] for item in trajectories], dtype=float
    )
    independent_rate = np.asarray(
        [item[1]["instantaneous_rate"] for item in trajectories], dtype=float
    )
    hierarchical_rate = np.asarray(
        [item[2]["instantaneous_rate"] for item in trajectories], dtype=float
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        [], [], color=independent_color, label="Independent (64 tiles)"
    )
    axes[0].plot([], [], color=hierarchical_color, label="Hierarchical (64 tiles)")
    for values in independent_thickness:
        axes[0].plot(days, values, color=independent_color, alpha=0.28, linewidth=0.8)
    for values in hierarchical_thickness:
        axes[0].plot(days, values, color=hierarchical_color, alpha=0.28, linewidth=0.8)
    axes[0].set_title("WFI01 ice thickness: all tiles")
    axes[0].set_ylabel("Ice thickness (nm)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    for values in independent_rate:
        axes[1].plot(days, values, color=independent_color, alpha=0.28, linewidth=0.8)
    for values in hierarchical_rate:
        axes[1].plot(days, values, color=hierarchical_color, alpha=0.28, linewidth=0.8)
    axes[1].set_title("WFI01 instantaneous rate: all tiles")
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("Rate (nm/day)")
    axes[1].grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_optical_comparison(path, records):
    tile_ids = np.asarray([record["tile_id"] for record in records])
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for key, label, axis in (
        ("hgcdte_n_log_scale", "HgCdTe n log scale", axes[0, 0]),
        ("hgcdte_k_log_scale", "HgCdTe k log scale", axes[0, 1]),
    ):
        independent = np.asarray([record[f"independent_{key}"] for record in records])
        hierarchical = np.asarray([record[f"hierarchical_{key}"] for record in records])
        axis.plot(tile_ids, independent, "o", color="tab:blue", markersize=3, label="Independent")
        axis.axhline(hierarchical[0], color="tab:orange", linewidth=1.8, label="Hierarchical shared")
        axis.set_xlabel("Tile ID")
        axis.set_ylabel(label)
        axis.legend(frameon=False)
    for layer in range(3):
        independent = np.asarray(
            [record[f"independent_ar_layer{layer + 1}_n"] for record in records]
        )
        hierarchical = np.asarray(
            [record[f"hierarchical_ar_layer{layer + 1}_n"] for record in records]
        )
        axes[1, 0].plot(tile_ids, independent, ".", label=f"Layer {layer + 1} independent")
        axes[1, 0].axhline(
            hierarchical[0], linestyle="--", linewidth=1.0, label=f"Layer {layer + 1} hierarchical"
        )
    axes[1, 0].set_title("AR refractive index n")
    axes[1, 0].set_xlabel("Tile ID")
    axes[1, 0].set_ylabel("n")
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)
    for layer in range(3):
        independent = np.asarray(
            [record[f"independent_ar_layer{layer + 1}_thickness"] for record in records]
        )
        hierarchical = np.asarray(
            [record[f"hierarchical_ar_layer{layer + 1}_thickness"] for record in records]
        )
        axes[1, 1].plot(tile_ids, independent, ".", label=f"Layer {layer + 1} independent")
        axes[1, 1].axhline(
            hierarchical[0], linestyle="--", linewidth=1.0, label=f"Layer {layer + 1} hierarchical"
        )
    axes[1, 1].set_title("AR thickness scale")
    axes[1, 1].set_xlabel("Tile ID")
    axes[1, 1].set_ylabel("Scale")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("WFI01 independent versus hierarchical optical properties")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_summary(path, records, trajectories, missing_tile_ids):
    ice_rms = np.asarray(
        [
            np.sqrt(np.mean((item[1]["ice_thickness"] - item[2]["ice_thickness"]) ** 2))
            for item in trajectories
        ]
    )
    n_delta = np.asarray(
        [record["independent_hgcdte_n_log_scale"] - record["hierarchical_hgcdte_n_log_scale"] for record in records]
    )
    k_delta = np.asarray(
        [record["independent_hgcdte_k_log_scale"] - record["hierarchical_hgcdte_k_log_scale"] for record in records]
    )
    lines = [
        "# WFI01 Independent versus Hierarchical Tile Fits",
        "",
        f"- Tiles compared: `{len(records)}`.",
        f"- Tiles not yet available: `{', '.join(map(str, missing_tile_ids)) or 'none'}`.",
        "- Each independent fit uses the same no-q absolute-substrate-power model and HgCdTe prior sigma 0.25.",
        "- Independent fit paths are taken from the supplied independent-fit root.",
        "",
        "## Differences",
        "",
        f"- RMS independent-minus-hierarchical ice-trajectory difference: `{np.mean(ice_rms):.6g}` nm mean across tiles; `{np.max(ice_rms):.6g}` nm maximum.",
        f"- Independent-minus-hierarchical HgCdTe n-log-scale difference: mean `{np.mean(n_delta):.6g}`, range `{np.min(n_delta):.6g}` to `{np.max(n_delta):.6g}`.",
        f"- Independent-minus-hierarchical HgCdTe k-log-scale difference: mean `{np.mean(k_delta):.6g}`, range `{np.min(k_delta):.6g}` to `{np.max(k_delta):.6g}`.",
        "",
        "See `independent_vs_hierarchical_parameters.csv`, `independent_vs_hierarchical_trajectory.csv`, `independent_vs_hierarchical_trajectories.png`, and `independent_vs_hierarchical_optics.png`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-dir", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--independent-root", type=Path, default=DEFAULT_INDEPENDENT_ROOT)
    parser.add_argument("--tile-zero-dir", type=Path, default=DEFAULT_TILE_ZERO)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.independent_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_paths = tile_fit_paths(args.independent_root, args.tile_zero_dir)
    available_tile_ids = [
        tile_id
        for tile_id, fit_path in fit_paths.items()
        if (fit_path / "posterior_samples.npz").is_file()
    ]
    missing_tile_ids = [tile_id for tile_id in range(64) if tile_id not in available_tile_ids]
    if not available_tile_ids:
        raise FileNotFoundError("No independent tile fits are available")
    records = []
    trajectories = []
    for tile_id in available_tile_ids:
        independent_path = fit_paths[tile_id]
        independent = archive_values(independent_path, tile_id if tile_id else 0)
        hierarchical = archive_values(args.combined_dir, tile_id)
        records.append(
            parameter_record(
                tile_id,
                independent,
                hierarchical,
                independent_path,
                args.combined_dir,
            )
        )
        trajectories.append((tile_id, independent, hierarchical))
    write_parameter_table(output_dir / "independent_vs_hierarchical_parameters.csv", records)
    write_trajectory_table(output_dir / "independent_vs_hierarchical_trajectory.csv", trajectories)
    plot_trajectory_comparison(output_dir / "independent_vs_hierarchical_trajectories.png", trajectories)
    plot_optical_comparison(output_dir / "independent_vs_hierarchical_optics.png", records)
    write_summary(
        output_dir / "independent_vs_hierarchical_summary.md",
        records,
        trajectories,
        missing_tile_ids,
    )
    print(f"Compared {len(available_tile_ids)} tiles; missing {len(missing_tile_ids)}")
    print(f"Wrote {output_dir / 'independent_vs_hierarchical_parameters.csv'}")
    print(f"Wrote {output_dir / 'independent_vs_hierarchical_trajectory.csv'}")
    print(f"Wrote {output_dir / 'independent_vs_hierarchical_trajectories.png'}")
    print(f"Wrote {output_dir / 'independent_vs_hierarchical_optics.png'}")
    print(f"Wrote {output_dir / 'independent_vs_hierarchical_summary.md'}")


if __name__ == "__main__":
    main()
