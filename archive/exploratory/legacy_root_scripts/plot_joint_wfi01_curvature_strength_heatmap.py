"""Plot the WFI01 in-sample ±0.5% rates by filter and curvature strength.

The matrix styling and cell annotations are copied from
``plot_joint_amber_filter_heatmaps.py``; only the row dimension is changed
from detectors to the WFI01 regularization settings.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import heatmap_car201_allstar_model_correction as workflow


ROOT = Path(__file__).resolve().parent / (
    "20261111/corrected_statistics_4h_thinned/production/wfi01/relative_response/"
    "experiments/car190_amber_fixed_ice_inverse_wavelength_curvature/sweep"
)
TABLE = ROOT / "amber_in_sample_filter_within_0p5.csv"
STRENGTH_ORDER = [
    "No added curvature",
    "Curvature weight 0.0001",
    "Curvature weight 0.001",
    "Curvature weight 0.01",
    "Reciprocal MaxEnt + curvature 0.01",
]


def plot_heatmap(table, row_order, filter_order, fraction_column, count_column,
                 total_column, title, output):
    matrix = np.full((len(row_order), len(filter_order)), np.nan)
    within = np.zeros_like(matrix, dtype=int)
    total = np.zeros_like(matrix, dtype=int)
    for _, row in table.iterrows():
        i = row_order.index(row["model"])
        j = filter_order.index(row["filter"])
        matrix[i, j] = float(row[fraction_column])
        within[i, j] = int(row[count_column])
        total[i, j] = int(row[total_column])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dedede")
    # Keep the existing heatmap cell height while sizing the four-row panel
    # compactly rather than leaving the 18-detector figure's unused whitespace.
    figure_height = max(3.8, 0.64 * len(row_order) + 1.8)
    fig, axis = plt.subplots(figsize=(12.5, figure_height))
    image = axis.imshow(np.ma.masked_invalid(matrix), origin="upper", aspect="auto",
                        vmin=0.0, vmax=1.0, cmap=cmap, interpolation="nearest")
    for i in range(len(row_order)):
        for j in range(len(filter_order)):
            if not np.isfinite(matrix[i, j]):
                axis.text(j, i, "—", ha="center", va="center", fontsize=8, color="0.4")
                continue
            color = "white" if matrix[i, j] < 0.55 else "black"
            axis.text(j, i, f"{100 * matrix[i, j]:.0f}%\n{within[i,j]}/{total[i,j]}",
                      ha="center", va="center", fontsize=7.2, color=color)
    axis.set_xticks(np.arange(len(filter_order)), labels=filter_order)
    axis.set_yticks(np.arange(len(row_order)), labels=row_order)
    axis.set_xlabel("CAR201 filter")
    axis.set_ylabel("Spectral modulation regularization")
    axis.set_xticks(np.arange(-0.5, len(filter_order), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    cbar.set_label("Comparisons within ±0.5%")
    fig.suptitle(title, fontsize=14, y=0.985)
    note = ("Cell labels: percent within ±0.5%; within-threshold count / eligible comparisons.\n"
            "Visit changes use measured EP3 pass 1 as reference; reference rows are excluded.")
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8.5, linespacing=1.25)
    fig.tight_layout(rect=(0.02, 0.18, 0.98, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--output-stem", type=Path,
                        default=ROOT / "amber_in_sample_curvature_strength_within_0p5")
    args = parser.parse_args()
    table = pd.read_csv(args.table)
    filters = list(workflow.FILTERS)
    present = list(table["model"].drop_duplicates())
    row_order = [label for label in STRENGTH_ORDER if label in present]
    row_order.extend(label for label in present if label not in row_order)
    expected_cells = len(row_order) * len(filters)
    if len(table) != expected_cells:
        raise ValueError(f"Expected {expected_cells} strength/filter cells, got {len(table)}")
    if set(table["model"]) != set(row_order):
        raise ValueError("The in-sample table has inconsistent regularization rows")

    output_png = args.output_stem.with_suffix(".png")
    output_pdf = args.output_stem.with_suffix(".pdf")
    figure = plot_heatmap(
        table,
        row_order,
        filters,
        "fraction_within_0p5_percent",
        "within_0p5_percent_count",
        "comparison_count",
        "WFI01 joint modulation: in-sample observations within ±0.5%",
        output_png,
    )
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_png} and {output_pdf}")


if __name__ == "__main__":
    main()
