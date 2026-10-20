"""Compare WFI production tile rates and residual RMS between two branches."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from plot_wfi_model_residual_map import load_residual_rows


RATE_FIELDS = (
    ("initial_rate_nm_day_median", "initial_rate_nm_day"),
    ("final_rate_nm_day_median", "final_rate_nm_day"),
    ("mean_rate_nm_day_median", "mean_rate_nm_day"),
)
RMS_FIELDS = (
    ("rms_residual_log", "rms_residual_log"),
    ("pre_rms_residual_log", "pre_rms_residual_log"),
    ("post_rms_residual_log", "post_rms_residual_log"),
)


def load_rate_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rate rows found in {path}")
    return {int(row["tile_id"]): row for row in rows}


def load_residual_dict(detector_root, tile_size):
    rows = load_residual_rows(detector_root, "WFI01", tile_size)
    return {int(row["tile_id"]): row for row in rows}


def make_comparison_rows(baseline_rates, candidate_rates, baseline_rms, candidate_rms):
    tile_ids = sorted(
        set(baseline_rates)
        & set(candidate_rates)
        & set(baseline_rms)
        & set(candidate_rms)
    )
    if not tile_ids:
        raise ValueError("No common WFI01 tiles found between branches")
    rows = []
    for tile_id in tile_ids:
        base_rate = baseline_rates[tile_id]
        new_rate = candidate_rates[tile_id]
        base_rms = baseline_rms[tile_id]
        new_rms = candidate_rms[tile_id]
        row = {
            "tile_id": tile_id,
            "tile_row": int(new_rate["tile_row"]),
            "tile_col": int(new_rate["tile_col"]),
        }
        for source_field, output_field in RATE_FIELDS:
            baseline = float(base_rate[source_field])
            candidate = float(new_rate[source_field])
            row[f"baseline_{output_field}"] = baseline
            row[f"candidate_{output_field}"] = candidate
            row[f"delta_{output_field}"] = candidate - baseline
        for source_field, output_field in RMS_FIELDS:
            baseline = float(base_rms[source_field])
            candidate = float(new_rms[source_field])
            row[f"baseline_{output_field}"] = baseline
            row[f"candidate_{output_field}"] = candidate
            row[f"delta_{output_field}"] = candidate - baseline
        rows.append(row)
    return rows


def tile_map(rows, field):
    size = max(max(row["tile_row"], row["tile_col"]) for row in rows) + 1
    values = np.full((size, size), np.nan)
    for row in rows:
        values[row["tile_row"], row["tile_col"]] = row[field]
    return values


def draw_difference_map(path, rows, baseline_label, candidate_label):
    rate_maps = [tile_map(rows, f"delta_{field}") for _source, field in RATE_FIELDS]
    rms_maps = [tile_map(rows, f"delta_{field}") * 1000.0 for _source, field in RMS_FIELDS]
    all_maps = rate_maps + rms_maps
    rate_finite = np.concatenate([values[np.isfinite(values)] for values in rate_maps])
    rms_finite = np.concatenate([values[np.isfinite(values)] for values in rms_maps])
    limits = (
        max(float(np.max(np.abs(rate_finite))), 1.0e-12),
        max(float(np.max(np.abs(rms_finite))), 1.0e-12),
    )

    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    titles = (
        "Initial rate delta (nm/day)",
        "Final rate delta (nm/day)",
        "Mean rate delta (nm/day)",
        "All-group RMS delta (10^-3 log)",
        "Pre-Decon2 RMS delta (10^-3 log)",
        "Post-Decon2 RMS delta (10^-3 log)",
    )
    for index, (axis, values, title) in enumerate(zip(axes.flat, all_maps, titles)):
        limit = limits[0 if index < len(rate_maps) else 1]
        image = axis.imshow(
            values,
            origin="lower",
            interpolation="none",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        axis.set_title(title)
        axis.set_xlabel("Tile column")
        axis.set_ylabel("Tile row")
        axis.set_xticks(range(values.shape[1]))
        axis.set_yticks(range(values.shape[0]))
        axis.grid(color="white", alpha=0.35, linewidth=0.5)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("WFI01 branch differences", fontsize=14)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_outputs(output_dir, rows, baseline_label, candidate_label):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    csv_path = output_dir / "wfi01_tile_branch_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "wfi01_tile_branch_difference_maps.png"
    draw_difference_map(plot_path, rows, baseline_label, candidate_label)

    lines = [
        f"# WFI01 production branch comparison",
        "",
        f"- Baseline: `{baseline_label}`.",
        f"- Candidate: `{candidate_label}`.",
        f"- Common tiles: `{len(rows)}`.",
        "- Rate deltas are candidate minus baseline in nm/day.",
        "- RMS deltas are candidate minus baseline in log units.",
        "",
    ]
    for field in tuple(f"delta_{field}" for _source, field in RATE_FIELDS) + tuple(
        f"delta_{field}" for _source, field in RMS_FIELDS
    ):
        values = np.asarray([row[field] for row in rows], dtype=float)
        lines.append(
            f"- `{field}`: min `{np.min(values):.6g}`, median `{np.median(values):.6g}`, "
            f"max `{np.max(values):.6g}`, mean `{np.mean(values):.6g}`."
        )
    (output_dir / "wfi01_tile_branch_comparison.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--baseline-label", default="n prior sigma = 0.25")
    parser.add_argument("--candidate-label", default="n prior sigma = 0.05")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    baseline_rates = load_rate_rows(args.baseline_dir / "tile_rate_posterior.csv")
    candidate_rates = load_rate_rows(args.candidate_dir / "tile_rate_posterior.csv")
    baseline_rms = load_residual_dict(args.baseline_dir, args.tile_size)
    candidate_rms = load_residual_dict(args.candidate_dir, args.tile_size)
    rows = make_comparison_rows(
        baseline_rates,
        candidate_rates,
        baseline_rms,
        candidate_rms,
    )
    write_outputs(
        args.output_dir,
        rows,
        args.baseline_label,
        args.candidate_label,
    )


if __name__ == "__main__":
    main()
