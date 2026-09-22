"""Compare tile-rate maps with default and relaxed neighbor regularization."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
SCAS = ("WFI01", "WFI02")
FIELDS = (
    ("initial_rate_nm_day_median", "Initial rate (nm/day)", "initial"),
    ("mean_rate_nm_day_median", "Mean rate (nm/day)", "mean"),
    ("final_rate_nm_day_median", "Final rate (nm/day)", "final"),
)


def read_rates(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise ValueError(f"Expected 64 tile rows in {path}, found {len(rows)}")
    return {
        int(row["tile_id"]): {
            "row": int(row["tile_row"]),
            "col": int(row["tile_col"]),
            **{
                field: float(row[field])
                for field, _, _ in FIELDS
            },
        }
        for row in rows
    }


def tile_grid(values, field):
    grid = np.full((8, 8), np.nan)
    for tile in values.values():
        grid[tile["row"], tile["col"]] = tile[field]
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"Incomplete tile grid for {field}")
    return grid


def adjacency(values):
    by_position = {(tile["row"], tile["col"]): tile_id for tile_id, tile in values.items()}
    edges = []
    for (row, col), tile_id in by_position.items():
        for position in ((row + 1, col), (row, col + 1)):
            if position in by_position:
                edges.append((tile_id, by_position[position]))
    return edges


def spread_metrics(values, relaxed_values, field):
    default = np.asarray([tile[field] for tile in values.values()])
    relaxed = np.asarray([tile[field] for tile in relaxed_values.values()])
    edges = adjacency(values)
    default_edges = np.asarray([values[a][field] - values[b][field] for a, b in edges])
    relaxed_edges = np.asarray(
        [relaxed_values[a][field] - relaxed_values[b][field] for a, b in edges]
    )
    return {
        "default_std": float(np.std(default, ddof=1)),
        "relaxed_std": float(np.std(relaxed, ddof=1)),
        "default_range": float(np.ptp(default)),
        "relaxed_range": float(np.ptp(relaxed)),
        "default_edge_rms": float(np.sqrt(np.mean(default_edges**2))),
        "relaxed_edge_rms": float(np.sqrt(np.mean(relaxed_edges**2))),
        "default_edge_max": float(np.max(np.abs(default_edges))),
        "relaxed_edge_max": float(np.max(np.abs(relaxed_edges))),
    }


def load_sca_pair(sca):
    default_path = ROOT / f"{sca.lower()}_rcs1039_full64_rate_5nm_np_ledacm_5000" / "tile_rate_posterior.csv"
    relaxed_path = ROOT / f"{sca.lower()}_rcs1039_full64_rate_5nm_np_ledacm_neighbor02_5000" / "tile_rate_posterior.csv"
    default = read_rates(default_path)
    relaxed = read_rates(relaxed_path)
    if set(default) != set(relaxed):
        raise ValueError(f"Tile IDs differ between default and relaxed fits for {sca}")
    return default, relaxed


def plot_map_comparison(path, field, label, pairs, metrics):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    cmap = "viridis"
    for row_index, sca in enumerate(SCAS):
        default, relaxed = pairs[sca]
        default_grid = tile_grid(default, field)
        relaxed_grid = tile_grid(relaxed, field)
        difference = relaxed_grid - default_grid
        limits = (min(np.min(default_grid), np.min(relaxed_grid)), max(np.max(default_grid), np.max(relaxed_grid)))
        difference_limit = max(abs(np.min(difference)), abs(np.max(difference)), 1.0e-9)
        images = (
            (default_grid, f"{sca} default\nstd={metrics[sca]['default_std']:.4g}"),
            (relaxed_grid, f"{sca} neighbor sigma=0.20\nstd={metrics[sca]['relaxed_std']:.4g}"),
            (
                difference,
                f"{sca} relaxed - default\nedge RMS: {metrics[sca]['relaxed_edge_rms']:.4g}",
            ),
        )
        for column, (grid, title) in enumerate(images):
            if column < 2:
                image = axes[row_index, column].imshow(
                    grid,
                    origin="upper",
                    cmap=cmap,
                    vmin=limits[0],
                    vmax=limits[1],
                )
            else:
                image = axes[row_index, column].imshow(
                    grid,
                    origin="upper",
                    cmap="coolwarm",
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-difference_limit, vmax=difference_limit),
                )
            axes[row_index, column].set_title(title)
            axes[row_index, column].set_xticks(range(8))
            axes[row_index, column].set_yticks(range(8))
            axes[row_index, column].set_xlabel("Tile column")
            axes[row_index, column].set_ylabel("Tile row")
            figure.colorbar(image, ax=axes[row_index, column], shrink=0.82)

        default_values = [tile[field] for tile in default.values()]
        relaxed_values = [tile[field] for tile in relaxed.values()]
        axes[row_index, 3].hist(default_values, bins=12, alpha=0.65, label="default")
        axes[row_index, 3].hist(relaxed_values, bins=12, alpha=0.65, label="sigma=0.20")
        axes[row_index, 3].set_title(
            f"{sca} tile distribution\n"
            f"edge RMS {metrics[sca]['default_edge_rms']:.4g} -> {metrics[sca]['relaxed_edge_rms']:.4g}"
        )
        axes[row_index, 3].set_xlabel(label)
        axes[row_index, 3].set_ylabel("Tile count")
        axes[row_index, 3].legend(frameon=False)
        axes[row_index, 3].grid(alpha=0.25)
    figure.suptitle(f"Tile-rate comparison: {label}")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_metrics(path, all_metrics):
    fields = [
        "sca",
        "rate_field",
        "default_std",
        "relaxed_std",
        "default_range",
        "relaxed_range",
        "default_edge_rms",
        "relaxed_edge_rms",
        "default_edge_max",
        "relaxed_edge_max",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "wfi_neighbor02_rate_comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = {sca: load_sca_pair(sca) for sca in SCAS}
    metrics_rows = []
    for field, label, short_name in FIELDS:
        metrics = {
            sca: spread_metrics(*pairs[sca], field)
            for sca in SCAS
        }
        output = args.output_prefix.with_name(
            f"{args.output_prefix.name}_{short_name}.png"
        )
        plot_map_comparison(output, field, label, pairs, metrics)
        for sca in SCAS:
            metrics_rows.append(
                {
                    "sca": sca,
                    "rate_field": short_name,
                    **metrics[sca],
                }
            )
        for sca in SCAS:
            print(
                f"{sca} {short_name}: "
                f"std {metrics[sca]['default_std']:.8g} -> {metrics[sca]['relaxed_std']:.8g}; "
                f"edge RMS {metrics[sca]['default_edge_rms']:.8g} -> "
                f"{metrics[sca]['relaxed_edge_rms']:.8g}"
            )
        print(f"Wrote {output}")
    metrics_path = args.output_prefix.with_suffix(".csv")
    write_metrics(metrics_path, metrics_rows)
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
