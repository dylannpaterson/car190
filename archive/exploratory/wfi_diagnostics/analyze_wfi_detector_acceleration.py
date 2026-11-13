"""Aggregate reset-aware tile acceleration diagnostics across one detector."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import run_wfi_production as production_runner
from analyze_wfi_tile_acceleration import (
    native_acceleration,
    segment_natural_log_rate_splines,
)
from fit_wfi_tmm_production import load_tile_statistics
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    positive_integer,
    tile_statistics_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", default="WFI01")
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    return parser.parse_args()


def load_tile_nodes(fit_dir):
    with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as samples:
        rates = np.asarray(samples["node_instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        thickness = np.asarray(samples["np_node_thickness"], dtype=float).reshape(-1)
    return rates, thickness


def linear_log_rate_trend(days, rates):
    log_rates = np.log(rates)
    slope, intercept = np.polyfit(days, log_rates, 1)
    fitted = intercept + slope * days
    residual = log_rates - fitted
    total = np.sum((log_rates - np.mean(log_rates)) ** 2)
    r_squared = 1.0 - np.sum(residual**2) / total if total > 0.0 else np.nan
    return float(slope), float(r_squared)


def segment_metrics(days, rates, thickness, reset_day):
    metrics = {}
    for segment, mask in (
        ("pre", days < reset_day),
        ("post", days >= reset_day),
    ):
        segment_days = days[mask]
        segment_rates = rates[mask]
        segment_thickness = thickness[mask]
        if len(segment_days) < 4:
            raise ValueError(f"{segment} segment has too few nodes")
        native_days = np.linspace(segment_days[0], segment_days[-1], 600)
        native = native_acceleration(native_days, segment_days, np.log(segment_rates))
        natural = segment_natural_log_rate_splines(
            segment_days, segment_rates, len(native_days)
        )
        strong = next(result for result in natural if result["label"] == "lambda=1e4")
        gcv = next(result for result in natural if result["label"] == "GCV")
        slope, r_squared = linear_log_rate_trend(segment_days, segment_rates)
        metrics.update(
            {
                f"{segment}_start_rate_nm_day": segment_rates[0],
                f"{segment}_end_rate_nm_day": segment_rates[-1],
                f"{segment}_linear_log_rate_slope_per_day": slope,
                f"{segment}_linear_log_rate_r_squared": r_squared,
                f"{segment}_native_median_acceleration_nm_day2": np.median(native),
                f"{segment}_natural_gcv_median_acceleration_nm_day2": np.median(gcv["acceleration"]),
                f"{segment}_natural_strong_median_acceleration_nm_day2": np.median(strong["acceleration"]),
            }
        )
    return metrics


def write_maps(path, rows, metrics, detector):
    n_rows = max(int(row["tile_row"]) for row in rows) + 1
    n_cols = max(int(row["tile_col"]) for row in rows) + 1
    figure, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    for axis, (label, field, scale) in zip(axes.flat, metrics):
        values = np.full((n_rows, n_cols), np.nan)
        for row in rows:
            values[int(row["tile_row"]), int(row["tile_col"])] = scale * float(row[field])
        finite = values[np.isfinite(values)]
        limit = max(abs(float(np.min(finite))), abs(float(np.max(finite))), 1.0e-12)
        image = axis.imshow(
            values,
            origin="lower",
            interpolation="none",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(label)
        axis.set_xlabel("Tile column")
        axis.set_ylabel("Tile row")
        axis.set_xticks(range(n_cols))
        axis.set_yticks(range(n_rows))
        axis.grid(color="white", alpha=0.25, linewidth=0.5)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{detector} tile acceleration and log-rate trend variation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_acceleration_distributions(path, rows, native_values, detector):
    """Plot pooled native interval and per-tile acceleration distributions."""
    colors = {"pre": "#0072B2", "post": "#D55E00"}
    labels = {"pre": "Pre-Decon2", "post": "Post-Decon2"}
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    pooled = np.concatenate(
        [np.asarray(native_values[segment], dtype=float) for segment in ("pre", "post")]
    )
    pooled_edges = np.histogram_bin_edges(pooled, bins=28)
    for segment in ("pre", "post"):
        values = np.asarray(native_values[segment], dtype=float)
        axes[0, 0].hist(
            values,
            bins=pooled_edges,
            density=True,
            alpha=0.55,
            color=colors[segment],
            label=f"{labels[segment]} intervals (n={values.size})",
        )
        axes[0, 0].axvline(
            np.median(values), color=colors[segment], linestyle="--", linewidth=1.5
        )
    axes[0, 0].set_title("Native interval acceleration")
    axes[0, 0].set_xlabel("d(rate)/dt (nm/day^2)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(frameon=False, fontsize=8)

    per_tile_fields = (
        ("native", "pre_native_median_acceleration_nm_day2", "Native median"),
        (
            "strong",
            "pre_natural_strong_median_acceleration_nm_day2",
            "Strong spline median",
        ),
    )
    for axis, (_, pre_field, title) in zip(axes[0, 1:], per_tile_fields):
        combined = np.asarray(
            [
                float(row[pre_field])
                for row in rows
            ]
            + [
                float(row[pre_field.replace("pre_", "post_")])
                for row in rows
            ],
            dtype=float,
        )
        edges = np.histogram_bin_edges(combined, bins=18)
        for segment, field in (
            ("pre", pre_field),
            ("post", pre_field.replace("pre_", "post_")),
        ):
            values = np.asarray([float(row[field]) for row in rows], dtype=float)
            axis.hist(
                values,
                bins=edges,
                density=True,
                alpha=0.55,
                color=colors[segment],
                label=labels[segment],
            )
            axis.axvline(
                np.median(values), color=colors[segment], linestyle="--", linewidth=1.5
            )
        axis.set_title(f"Per-tile {title.lower()} acceleration")
        axis.set_xlabel("Median d(rate)/dt (nm/day^2)")
        axis.set_ylabel("Density")
        axis.legend(frameon=False, fontsize=8)

    post_field = "post_natural_strong_median_acceleration_nm_day2"
    row_values = np.asarray([int(row["tile_row"]) for row in rows], dtype=int)
    col_values = np.asarray([int(row["tile_col"]) for row in rows], dtype=int)
    post_values = np.asarray([float(row[post_field]) for row in rows], dtype=float)
    scatter = axes[1, 0].scatter(
        col_values,
        post_values,
        c=row_values,
        cmap="viridis",
        s=42,
        edgecolor="black",
        linewidth=0.25,
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Post-Decon2 strong-spline acceleration by column")
    axes[1, 0].set_xlabel("Tile column")
    axes[1, 0].set_ylabel("Median d(rate)/dt (nm/day^2)")
    axes[1, 0].set_xticks(sorted(set(col_values)))
    figure.colorbar(scatter, ax=axes[1, 0], label="Tile row")

    for segment in ("pre", "post"):
        field = f"{segment}_natural_strong_median_acceleration_nm_day2"
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        axes[1, 1].scatter(
            [float(row[f"{segment}_linear_log_rate_slope_per_day"]) for row in rows],
            values,
            color=colors[segment],
            alpha=0.7,
            label=labels[segment],
        )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Smoothed acceleration versus log-rate trend")
    axes[1, 1].set_xlabel("Linear log-rate slope (1/day)")
    axes[1, 1].set_ylabel("Strong-spline median acceleration (nm/day^2)")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for segment in ("pre", "post"):
        native_field = f"{segment}_native_median_acceleration_nm_day2"
        strong_field = f"{segment}_natural_strong_median_acceleration_nm_day2"
        axes[1, 2].scatter(
            [float(row[native_field]) for row in rows],
            [float(row[strong_field]) for row in rows],
            color=colors[segment],
            alpha=0.7,
            label=labels[segment],
        )
    combined_medians = np.asarray(
        [
            float(row[field])
            for row in rows
            for field in (
                "pre_native_median_acceleration_nm_day2",
                "post_native_median_acceleration_nm_day2",
                "pre_natural_strong_median_acceleration_nm_day2",
                "post_natural_strong_median_acceleration_nm_day2",
            )
        ],
        dtype=float,
    )
    axes[1, 2].plot(
        [np.min(combined_medians), np.max(combined_medians)],
        [np.min(combined_medians), np.max(combined_medians)],
        color="0.35",
        linestyle=":",
        linewidth=1.0,
        label="1:1",
    )
    axes[1, 2].set_title("Native versus strong-spline tile medians")
    axes[1, 2].set_xlabel("Native median acceleration (nm/day^2)")
    axes[1, 2].set_ylabel("Strong-spline median acceleration (nm/day^2)")
    axes[1, 2].legend(frameon=False, fontsize=8)

    figure.suptitle(f"{detector} acceleration distributions and spatial structure")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector_root = args.detector_root
    detector_prefix = args.detector.lower()
    rows = []
    native_values = {"pre": [], "post": []}
    for fit_dir in sorted((detector_root / "fits").glob("tile_*")):
        tile_id = int(fit_dir.name.split("_")[1])
        statistics_path = tile_statistics_path(
            detector_root / "tile_statistics", detector, args.tile_size, tile_id
        )
        if not production_runner.archive_matches_statistics(fit_dir, statistics_path):
            continue
        data = load_tile_statistics(statistics_path)
        rates, thickness = load_tile_nodes(fit_dir)
        metrics = segment_metrics(
            np.asarray(data["node_days"], dtype=float),
            rates,
            thickness,
            float(data["reset_day"]),
        )
        for segment, mask in (
            ("pre", np.asarray(data["node_days"]) < float(data["reset_day"])),
            ("post", np.asarray(data["node_days"]) >= float(data["reset_day"])),
        ):
            segment_days = np.asarray(data["node_days"], dtype=float)[mask]
            segment_rates = rates[mask]
            interval_days = 0.5 * (segment_days[:-1] + segment_days[1:])
            native_values[segment].extend(
                native_acceleration(
                    interval_days, segment_days, np.log(segment_rates)
                ).tolist()
            )
        tile_metadata = data["metadata"][int(tile_id)]
        row = {
            "detector": args.detector,
            "tile_id": int(tile_id),
            "tile_row": int(tile_metadata["tile_row"]),
            "tile_col": int(tile_metadata["tile_col"]),
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise ValueError("No matching tile fits found")
    rows.sort(key=lambda row: int(row["tile_id"]))
    fields = list(rows[0])
    with (args.output_dir / f"{detector_prefix}_tile_acceleration_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    map_metrics = (
        ("Pre-Decon2 linear log-rate slope (1/day)", "pre_linear_log_rate_slope_per_day", 1.0),
        ("Post-Decon2 linear log-rate slope (1/day)", "post_linear_log_rate_slope_per_day", 1.0),
        ("Pre-Decon2 native median acceleration (nm/day^2)", "pre_native_median_acceleration_nm_day2", 1.0),
        ("Post-Decon2 native median acceleration (nm/day^2)", "post_native_median_acceleration_nm_day2", 1.0),
        ("Pre-Decon2 strong natural spline (nm/day^2)", "pre_natural_strong_median_acceleration_nm_day2", 1.0),
        ("Post-Decon2 strong natural spline (nm/day^2)", "post_natural_strong_median_acceleration_nm_day2", 1.0),
    )
    write_maps(
        args.output_dir / f"{detector_prefix}_tile_acceleration_maps.png",
        rows,
        map_metrics,
        args.detector,
    )
    write_acceleration_distributions(
        args.output_dir / f"{detector_prefix}_acceleration_distributions.png",
        rows,
        native_values,
        args.detector,
    )

    lines = [
        f"# {args.detector} tile acceleration variation",
        "",
        f"- Matching tiles: `{len(rows)}`.",
        "- All values come from reset-separated post-processing of the tile MAP fits.",
        "- Strong natural-spline acceleration is regularized and should be read as a common trend diagnostic, not a posterior measurement.",
        "",
    ]
    for label, field, _ in map_metrics:
        values = np.asarray([float(row[field]) for row in rows])
        row_index = np.asarray([int(row["tile_row"]) for row in rows])
        col_index = np.asarray([int(row["tile_col"]) for row in rows])
        row_corr = np.corrcoef(values, row_index)[0, 1] if np.std(values) else np.nan
        col_corr = np.corrcoef(values, col_index)[0, 1] if np.std(values) else np.nan
        lines.append(
            f"- {label}: min `{np.min(values):.6g}`, median `{np.median(values):.6g}`, "
            f"max `{np.max(values):.6g}`, std `{np.std(values):.6g}`, "
            f"row correlation `{row_corr:.4g}`, column correlation `{col_corr:.4g}`."
        )
    (args.output_dir / f"{detector_prefix}_tile_acceleration_summary.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(f"Matching tiles: {len(rows)}")
    print(f"Wrote {args.output_dir / f'{detector_prefix}_tile_acceleration_maps.png'}")
    print(f"Wrote {args.output_dir / f'{detector_prefix}_tile_acceleration_summary.csv'}")
    print(f"Wrote {args.output_dir / f'{detector_prefix}_tile_acceleration_summary.md'}")
    print(
        f"Wrote {args.output_dir / f'{detector_prefix}_acceleration_distributions.png'}"
    )


if __name__ == "__main__":
    main()
