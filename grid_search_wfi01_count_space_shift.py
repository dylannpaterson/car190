"""Grid-search a count-space relative time shift and profile grey scale."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from fit_wfi_tmm_numpyro import apply_fixed_corrections, load_tile_statistics


RESET_DAY = 15.064405023148149
SHIFT_STEP_DAYS = 0.0015063


def corrected_count_means(data, correction_path):
    corrected = apply_fixed_corrections(data, correction_path)
    if corrected.get("fixed_correction_space") != "counts":
        raise ValueError("The correction table must be marked as count-space")
    return np.asarray(corrected["sum_current_counts"], dtype=float) / np.asarray(
        data["counts"], dtype=float
    )


def _shift_arrays(data, values, reset_day, delta_t_from_t0):
    days = np.asarray(data["days"], dtype=float)
    leds = np.asarray(data["led"], dtype=int)
    pair_counts = np.asarray(data["counts"], dtype=float)
    pre_mask = days < reset_day
    post_mask = ~pre_mask
    shift_days = float(delta_t_from_t0 - reset_day)
    observed = []
    predicted = []
    weights = []
    valid_groups = 0

    for led in range(1, 7):
        pre_indices = np.flatnonzero(pre_mask & (leds == led))
        post_indices = np.flatnonzero(post_mask & (leds == led))
        if not pre_indices.size or not post_indices.size:
            continue
        shifted_days = days[post_indices] + shift_days
        interpolated = np.asarray(
            [
                np.interp(
                    shifted_days,
                    days[pre_indices],
                    values[tile_index, pre_indices],
                    left=np.nan,
                    right=np.nan,
                )
                for tile_index in range(values.shape[0])
            ]
        )
        valid = np.all(np.isfinite(interpolated), axis=0)
        valid_groups += int(np.count_nonzero(valid))
        if not np.any(valid):
            continue
        observed.append(values[:, post_indices][:, valid].ravel())
        predicted.append(interpolated[:, valid].ravel())
        weights.append(pair_counts[:, post_indices][:, valid].ravel())

    if not observed:
        observed = predicted = weights = None
    else:
        observed = np.concatenate(observed)
        predicted = np.concatenate(predicted)
        weights = np.concatenate(weights)
    total_post_groups = int(np.count_nonzero(post_mask))
    return {
        "delta_t_from_t0_days": float(delta_t_from_t0),
        "shift_days": shift_days,
        "valid_post_groups": valid_groups,
        "valid_post_points": 0 if observed is None else int(observed.size),
        "total_post_groups": total_post_groups,
        "total_post_points": int(values.shape[0] * total_post_groups),
        "observed": observed,
        "predicted": predicted,
        "weights": weights,
    }


def fit_shift(data, values, reset_day, delta_t_from_t0):
    arrays = _shift_arrays(data, values, reset_day, delta_t_from_t0)
    observed = arrays.pop("observed")
    predicted = arrays.pop("predicted")
    weights = arrays.pop("weights")
    if observed is None:
        return {
            **arrays,
            "raw_rms_counts": np.nan,
            "multiplicative_rms_counts": np.nan,
            "grey_scale": np.nan,
            "grey_offset_fraction": np.nan,
        }
    grey_scale = float(
        np.sum(weights * predicted * observed) / np.sum(weights * predicted * predicted)
    )
    raw_residual = observed - predicted
    fitted_residual = observed - grey_scale * predicted
    weight_sum = np.sum(weights)
    return {
        **arrays,
        "raw_rms_counts": float(np.sqrt(np.sum(weights * raw_residual**2) / weight_sum)),
        "multiplicative_rms_counts": float(
            np.sqrt(np.sum(weights * fitted_residual**2) / weight_sum)
        ),
        "grey_scale": grey_scale,
        "grey_offset_fraction": grey_scale - 1.0,
    }


def fit_joint_shift_scale(
    data,
    values,
    reset_day,
    initial,
    bounds,
    minimum_valid_groups,
):
    def objective(parameters):
        delta_t_from_t0, grey_scale = np.asarray(parameters, dtype=float)
        if not np.isfinite(delta_t_from_t0) or not np.isfinite(grey_scale):
            return 1.0e100
        arrays = _shift_arrays(data, values, reset_day, delta_t_from_t0)
        if (
            arrays["observed"] is None
            or arrays["valid_post_groups"] < minimum_valid_groups
            or grey_scale <= 0.0
        ):
            return 1.0e100
        residual = arrays["observed"] - grey_scale * arrays["predicted"]
        return float(np.sum(arrays["weights"] * residual**2))

    result = minimize(
        objective,
        np.asarray(initial, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1.0e-15, "gtol": 1.0e-10, "maxiter": 10000, "maxls": 50},
    )
    delta_t_from_t0, grey_scale = np.asarray(result.x, dtype=float)
    arrays = _shift_arrays(data, values, reset_day, delta_t_from_t0)
    if arrays["observed"] is None:
        raise RuntimeError("Joint shift/scale optimization found no valid overlap")
    residual = arrays["observed"] - grey_scale * arrays["predicted"]
    raw_residual = arrays["observed"] - arrays["predicted"]
    weight_sum = np.sum(arrays["weights"])
    return {
        "delta_t_from_t0_days": float(delta_t_from_t0),
        "shift_days": float(delta_t_from_t0 - reset_day),
        "valid_post_groups": arrays["valid_post_groups"],
        "valid_post_points": arrays["valid_post_points"],
        "total_post_groups": arrays["total_post_groups"],
        "total_post_points": arrays["total_post_points"],
        "raw_rms_counts": float(
            np.sqrt(np.sum(arrays["weights"] * raw_residual**2) / weight_sum)
        ),
        "multiplicative_rms_counts": float(
            np.sqrt(np.sum(arrays["weights"] * residual**2) / weight_sum)
        ),
        "grey_scale": float(grey_scale),
        "grey_offset_fraction": float(grey_scale - 1.0),
        "weighted_sse": float(result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(getattr(result, "nit", 0)),
        "optimizer_function_evaluations": int(getattr(result, "nfev", 0)),
    }


def write_grid(path, rows):
    fields = tuple(rows[0])
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def best_row(rows, predicate=lambda row: True):
    eligible = [
        row
        for row in rows
        if predicate(row) and np.isfinite(row["multiplicative_rms_counts"])
    ]
    return min(eligible, key=lambda row: row["multiplicative_rms_counts"]) if eligible else None


def write_plot(path, rows, reference_delta_t=None):
    delta_t = np.asarray([row["delta_t_from_t0_days"] for row in rows])
    rms = np.asarray([row["multiplicative_rms_counts"] for row in rows])
    valid_groups = np.asarray([row["valid_post_groups"] for row in rows])
    grey_percent = 100.0 * np.asarray(
        [row["grey_offset_fraction"] for row in rows]
    )
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(delta_t, rms, color="tab:blue", linewidth=1.1)
    axes[0].axvline(0.0, color="tab:red", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("Multiplicative RMS (counts)")
    current_index = None
    if reference_delta_t is not None:
        current_index = int(np.nanargmin(np.abs(delta_t - reference_delta_t)))
        current_grey_percent = grey_percent[current_index]
        axes[0].axvline(
            reference_delta_t,
            color="0.25",
            linestyle=":",
            linewidth=1.0,
            label=(
                f"reference = {reference_delta_t:.3f} d, "
                f"grey = {current_grey_percent:+.3f}%"
            ),
        )
        axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(delta_t, valid_groups, color="tab:green", linewidth=1.1)
    axes[1].axvline(0.0, color="tab:red", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("delta_t from decon1 t=0 (days)")
    axes[1].set_ylabel("Overlapping post groups")
    axes[1].grid(alpha=0.25)
    axes[2].plot(delta_t, grey_percent, color="tab:purple", linewidth=1.1)
    axes[2].axhline(0.0, color="0.35", linewidth=0.8)
    axes[2].axvline(0.0, color="tab:red", linestyle="--", linewidth=0.9)
    if current_index is not None:
        axes[2].axvline(
            reference_delta_t,
            color="0.25",
            linestyle=":",
            linewidth=1.0,
        )
        axes[2].plot(
            delta_t[current_index],
            current_grey_percent,
            color="0.25",
            marker="o",
            markersize=4,
        )
        axes[2].annotate(
            f"{current_grey_percent:+.3f}%",
            xy=(delta_t[current_index], current_grey_percent),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
            color="0.25",
        )
    axes[2].set_xlabel("delta_t from decon1 t=0 (days)")
    axes[2].set_ylabel("Grey correction (%)")
    axes[2].grid(alpha=0.25)
    figure.suptitle("Count-space shift search and grey correction")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reset-day", type=float, default=RESET_DAY)
    parser.add_argument("--additional-acm-step-repeat", type=int, default=39)
    parser.add_argument("--pre-reset-acm-step-repeat", type=int, default=35)
    parser.add_argument("--reference-delta-t-from-t0", type=float, default=None)
    parser.add_argument("--delta-t-min", type=float, default=None)
    parser.add_argument("--delta-t-max", type=float, default=None)
    parser.add_argument("--step-days", type=float, default=SHIFT_STEP_DAYS)
    args = parser.parse_args()
    if args.step_days <= 0.0 or not np.isfinite(args.step_days):
        raise ValueError("--step-days must be positive and finite")

    data = load_tile_statistics(
        args.statistics,
        additional_acm_step_repeat=args.additional_acm_step_repeat,
        pre_reset_acm_step_repeat=args.pre_reset_acm_step_repeat,
    )
    values = corrected_count_means(data, args.corrections)
    days = np.asarray(data["days"], dtype=float)
    post_days = days[days >= args.reset_day]
    delta_t_min = (
        args.reset_day - float(np.max(post_days))
        if args.delta_t_min is None
        else args.delta_t_min
    )
    delta_t_max = args.reset_day if args.delta_t_max is None else args.delta_t_max
    if delta_t_min >= delta_t_max:
        raise ValueError("delta-t range must be increasing")
    delta_grid = np.arange(
        delta_t_min,
        delta_t_max + 0.5 * args.step_days,
        args.step_days,
        dtype=float,
    )
    rows = [fit_shift(data, values, args.reset_day, delta_t) for delta_t in delta_grid]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_grid(args.output_dir / "extended_shift_grid.csv", rows)
    full_overlap_groups = max(row["valid_post_groups"] for row in rows)
    full_overlap_rows = [
        row for row in rows if row["valid_post_groups"] == full_overlap_groups
    ]
    best_full_overlap = best_row(full_overlap_rows)
    if best_full_overlap is None:
        raise RuntimeError("The shift grid contains no fully overlapping solution")
    joint_delta_bounds = (
        min(row["delta_t_from_t0_days"] for row in full_overlap_rows),
        max(row["delta_t_from_t0_days"] for row in full_overlap_rows),
    )
    joint_fit = fit_joint_shift_scale(
        data,
        values,
        args.reset_day,
        initial=(
            best_full_overlap["delta_t_from_t0_days"],
            best_full_overlap["grey_scale"],
        ),
        bounds=(joint_delta_bounds, (1.0e-6, None)),
        minimum_valid_groups=full_overlap_groups,
    )
    negative_threshold = int(np.ceil(0.75 * full_overlap_groups))
    summary = {
        "reset_day": args.reset_day,
        "delta_t_definition": "delta_t_from_t0 = reset_day + shift_days",
        "delta_t_min": float(delta_t_min),
        "delta_t_max": float(delta_t_max),
        "step_days": args.step_days,
        "full_overlap_post_groups": int(full_overlap_groups),
        "negative_selection_min_post_groups": negative_threshold,
        "reference_delta_t_from_t0_days": args.reference_delta_t_from_t0,
        "best_all": best_row(rows),
        "best_full_overlap_grid": best_full_overlap,
        "joint_fit_bounds": {
            "delta_t_from_t0_days": list(joint_delta_bounds),
            "grey_scale": [1.0e-6, None],
        },
        "joint_fit": joint_fit,
        "best_negative": best_row(rows, lambda row: row["delta_t_from_t0_days"] < 0.0),
        "best_negative_at_least_75_percent_overlap": best_row(
            rows,
            lambda row: row["delta_t_from_t0_days"] < 0.0
            and row["valid_post_groups"] >= negative_threshold,
        ),
        "best_nonnegative": best_row(
            rows, lambda row: row["delta_t_from_t0_days"] >= 0.0
        ),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    write_plot(
        args.output_dir / "extended_shift_grid.png",
        rows,
        args.reference_delta_t_from_t0,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
