"""Plot in-sample and direct-SED out-of-sample ±0.5% rates by filter."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import heatmap_car201_allstar_model_correction as workflow


ROOT = workflow.PRODUCTION_ROOT / (
    "wfi01/spline_relative_response_normal/experiments/"
    "car190_amber_joint_2day_all_detectors_overview"
)
TABLE = ROOT / "amber_in_vs_out_sample_within_0p5_by_filter.csv"


def plot_heatmap(table, detector_order, filter_order, fraction_column, count_column,
                 total_column, title, output, out_of_sample=False):
    matrix = np.full((len(detector_order), len(filter_order)), np.nan)
    within = np.zeros_like(matrix, dtype=int)
    total = np.zeros_like(matrix, dtype=int)
    for _, row in table.iterrows():
        i = detector_order.index(row["detector"])
        j = filter_order.index(row["filter"])
        matrix[i, j] = float(row[fraction_column])
        within[i, j] = int(row[count_column])
        total[i, j] = int(row[total_column])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dedede")
    fig, axis = plt.subplots(figsize=(12.5, 11.5))
    image = axis.imshow(np.ma.masked_invalid(matrix), origin="upper", aspect="auto",
                        vmin=0.0, vmax=1.0, cmap=cmap, interpolation="nearest")
    for i in range(len(detector_order)):
        for j in range(len(filter_order)):
            if not np.isfinite(matrix[i, j]):
                axis.text(j, i, "—", ha="center", va="center", fontsize=8, color="0.4")
                continue
            color = "white" if matrix[i, j] < 0.55 else "black"
            axis.text(j, i, f"{100 * matrix[i, j]:.0f}%\n{within[i,j]}/{total[i,j]}",
                      ha="center", va="center", fontsize=7.2, color=color)
    axis.set_xticks(np.arange(len(filter_order)), labels=filter_order)
    axis.set_yticks(np.arange(len(detector_order)), labels=detector_order)
    axis.set_xlabel("CAR201 filter")
    axis.set_ylabel("Roman detector")
    axis.set_xticks(np.arange(-0.5, len(filter_order), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(detector_order), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    cbar.set_label("Comparisons within ±0.5%")
    fig.suptitle(title, fontsize=14, y=0.985)
    note = ("Cell labels: percent within ±0.5%; within-threshold count / eligible comparisons.\n"
            "Visit changes use measured EP3 pass 1 as reference; reference rows are excluded.")
    if out_of_sample:
        note += "\nHeld-out stars with direct SEDs only; NN-imputed SEDs are reported separately."
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8.5, linespacing=1.25)
    fig.tight_layout(rect=(0.02, 0.13, 0.98, 0.95))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig


def main():
    table = pd.read_csv(TABLE)
    detectors = list(workflow.DETECTORS)
    filters = list(workflow.FILTERS)
    if len(table) != len(detectors) * len(filters):
        raise ValueError(f"Expected {len(detectors) * len(filters)} detector/filter cells, got {len(table)}")
    in_figure = plot_heatmap(
        table, detectors, filters,
        "fraction_within_0p5_percent", "within_0p5_percent_count", "comparison_count",
        "Joint modulated correction: in-sample observations within ±0.5%",
        ROOT / "amber_in_sample_within_0p5_by_filter.png",
    )
    oos_figure = plot_heatmap(
        table, detectors, filters,
        "out_of_sample_fraction_within_0p5_percent", "out_of_sample_within_0p5_percent_count",
        "out_of_sample_comparison_count",
        "Joint modulated correction: held-out direct-SED observations within ±0.5%",
        ROOT / "amber_out_of_sample_within_0p5_by_filter.png",
        out_of_sample=True,
    )
    with PdfPages(ROOT / "amber_filter_within_0p5_heatmaps.pdf") as pdf:
        pdf.savefig(in_figure)
        pdf.savefig(oos_figure)
    plt.close(in_figure)
    plt.close(oos_figure)

    uncorrected = pd.read_csv(ROOT / "amber_in_vs_out_sample_uncorrected_within_0p5_by_filter.csv")
    in_uncorrected = plot_heatmap(
        uncorrected, detectors, filters,
        "in_sample_fraction_within_0p5_percent", "in_sample_within_0p5_percent_count",
        "in_sample_comparison_count",
        "Roll-adjusted data (no ice correction): in-sample observations within ±0.5%",
        ROOT / "amber_in_sample_uncorrected_within_0p5_by_filter.png",
    )
    oos_uncorrected = plot_heatmap(
        uncorrected, detectors, filters,
        "out_of_sample_fraction_within_0p5_percent", "out_of_sample_within_0p5_percent_count",
        "out_of_sample_comparison_count",
        "Roll-adjusted data (no ice correction): held-out observations within ±0.5%",
        ROOT / "amber_out_of_sample_uncorrected_within_0p5_by_filter.png",
        out_of_sample=True,
    )
    with PdfPages(ROOT / "amber_uncorrected_filter_within_0p5_heatmaps.pdf") as pdf:
        pdf.savefig(in_uncorrected)
        pdf.savefig(oos_uncorrected)
    plt.close(in_uncorrected)
    plt.close(oos_uncorrected)
    print(f"Wrote corrected and uncorrected filter heatmaps under {ROOT}")


if __name__ == "__main__":
    main()
