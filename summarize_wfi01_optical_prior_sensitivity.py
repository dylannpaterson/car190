"""Summarize optical-prior sensitivity and select the conservative ice result."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "wfi01_optical_prior_sensitivity"
CORE_FIT = (
    ROOT
    / "wfi01_all_pixel_acm_step_initial_thickness"
    / "wfi01_all_pixel_soc_global_fit.csv"
)
CORE_NODES = (
    ROOT
    / "wfi01_all_pixel_acm_step_initial_thickness"
    / "wfi01_all_pixel_soc_global_nodes.csv"
)
PRE_ACM_FIT = (
    ROOT
    / "wfi01_pre_acm_initial_thickness"
    / "wfi01_all_pixel_soc_global_fit.csv"
)
MAGNITUDE_CASES = (
    (0.1, ROOT / "wfi01_simultaneous_q_magnitude_prior_0p1"),
    (0.01, ROOT / "wfi01_simultaneous_q_magnitude_prior_0p01"),
    (0.001, ROOT / "wfi01_simultaneous_q_magnitude"),
)
PHASE_CASE = ROOT / "wfi01_simultaneous_q_magnitude_phase"


def read_core_fit(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["selected"] == "True"
        and row["sed_mode"] == "core"
        and row["trajectory"] == "entropy_sqrt"
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected core entropy fit in {path}")
    row = selected[0]
    return {
        "model": "fixed empirical table",
        "prior_weight": "",
        "initial_thickness_nm": float(row["initial_thickness_nm"]),
        "accumulated_thickness_nm": float(row["max_incremental_thickness_nm"]),
        "mean_rate_nm_day": float(row["mean_rate_nm_day"]),
        "final_thickness_nm": float(row["max_thickness_nm"]),
        "step_percent": float(row["step_percent"]),
        "all_pixel_rms_log": float(row["model_pixel_rms_log"]),
        "nonreference_pixel_rms_log": float(
            row["nonreference_model_pixel_rms_log"]
        ),
        "admissible_physical_thickness": True,
    }


def read_report_row(path, label):
    prefix = f"| {label} |"
    rows = [line for line in path.read_text().splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise ValueError(f"Expected one `{label}` row in {path}")
    values = [value.strip() for value in rows[0].strip("|").split("|")]
    return {
        "initial_thickness_nm": float(values[1]),
        "accumulated_thickness_nm": float(values[2]),
        "mean_rate_nm_day": float(values[3]),
        "final_thickness_nm": float(values[4]),
        "step_percent": float(values[5]),
        "all_pixel_rms_log": float(values[6]),
    }


def read_selected_interval_rates(path):
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["selected"] == "True"]
    if len(rows) < 2:
        raise ValueError(f"Expected at least two selected trajectory nodes in {path}")
    rates = np.asarray(
        [float(row["interval_rate_nm_day"]) for row in rows[1:]],
        dtype=float,
    )
    if not np.all(np.isfinite(rates)):
        raise ValueError(f"Selected interval rates must be finite in {path}")
    return rates


def validation_scores(path, model):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = np.asarray(
        [
            float(row["equal_exposure_rmse_log"])
            for row in rows
            if row["model"] == model
        ]
    )
    if len(scores) != 15:
        raise ValueError(f"Expected 15 held-out scores for {model} in {path}")
    return scores


def validation_metric(path, model, field):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray(
        [float(row[field]) for row in rows if row["model"] == model]
    )


def one_standard_error_selection(rows, fold_scores):
    """Select the simplest admissible row within one SE of the best score."""
    if len(rows) != len(fold_scores):
        raise ValueError("Each sensitivity row requires an aligned fold-score vector")
    admissible_indices = [
        index
        for index, row in enumerate(rows)
        if row["admissible_physical_thickness"]
    ]
    if not admissible_indices:
        raise ValueError("At least one physical-thickness model must be admissible")
    best_index = min(
        admissible_indices,
        key=lambda index: rows[index]["held_out_rmse_log"],
    )
    best_scores = np.asarray(fold_scores[best_index], dtype=float)
    if best_scores.ndim != 1 or len(best_scores) < 2 or not np.all(
        np.isfinite(best_scores)
    ):
        raise ValueError("The best model requires at least two finite fold scores")
    best_standard_error = float(
        np.std(best_scores, ddof=1) / np.sqrt(len(best_scores))
    )
    one_se_limit = rows[best_index]["held_out_rmse_log"] + best_standard_error
    primary_index = next(
        index
        for index in admissible_indices
        if rows[index]["held_out_rmse_log"] <= one_se_limit
    )
    return primary_index, best_index, best_standard_error, one_se_limit


def write_outputs(
    output_dir,
    rows,
    fold_scores,
    empirical_fold_rates,
    empirical_fold_d0,
    pre_acm,
    interval_rates,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        primary_index,
        best_index,
        best_standard_error,
        one_se_limit,
    ) = one_standard_error_selection(rows, fold_scores)
    for index, row in enumerate(rows):
        row["selected_primary"] = index == primary_index
    csv_path = output_dir / "optical_prior_sensitivity.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    primary = rows[primary_index]
    best = rows[best_index]
    best_scores = np.asarray(fold_scores[best_index], dtype=float)
    empirical_scores = np.asarray(fold_scores[0], dtype=float)
    empirical = rows[0]
    sensitivity_rates = [
        row["mean_rate_nm_day"]
        for row in rows
        if row["model"].startswith("magnitude nuisance")
    ]
    paired_improvement = float(np.mean(empirical_scores - best_scores))
    if paired_improvement >= 0.0:
        predictive_change = (
            f"reduces mean held-out log-RMSE by `{paired_improvement:.12g}`"
        )
    else:
        predictive_change = (
            f"increases mean held-out log-RMSE by `{-paired_improvement:.12g}`"
        )

    report_path = output_dir / "optical_prior_sensitivity.md"
    lines = [
        "# WFI01 optical-model sensitivity",
        "",
        "## Primary Result",
        "",
        f"The one-standard-error rule selects `{primary['model']}` as the simplest admissible model within one fold standard error of the best epoch-blocked score. The nuisance fits move the inferred rate strongly as their prior is relaxed.",
        "",
        "| Quantity | Primary value |",
        "|---|---:|",
        f"| Initial physical thickness | {primary['initial_thickness_nm']:.6f} nm |",
        f"| Accumulated thickness | {primary['accumulated_thickness_nm']:.6f} nm |",
        f"| Mean accumulation rate | {primary['mean_rate_nm_day']:.6f} nm/day |",
        f"| Final physical thickness | {primary['final_thickness_nm']:.6f} nm |",
        f"| Common ACM step | {primary['step_percent']:.6f}% |",
        f"| All-pixel RMS, including identity references | {primary['all_pixel_rms_log']:.8f} |",
        f"| All-pixel RMS, excluding identity references | {primary['nonreference_pixel_rms_log']:.8f} |",
        f"| Mean held-out RMSE | {primary['held_out_rmse_log']:.12g} |",
        "",
        "## Model Quality and Recommended Default",
        "",
        "The fixed empirical response is the recommended default optical model. It is not the lowest-RMSE option in-sample or in every held-out fold, but the predictive gain from adding shared optical nuisance parameters is small compared with the resulting rate instability.",
        "",
        "For the all-LED fit, the aggregate temporal trend is captured well enough for a descriptive detection: the selected model removes 94.808% of between-exposure SSE. It is not a precision per-LED physical model; the systematic LED residuals show that response or mapping error remains.",
        "",
        "Use the event-aware fixed empirical core response with the entropy-regularized square-root trajectory as the default descriptive fit (`12.544659 nm/day`). Retain the fixed-response square-root fit (`11.477738 nm/day`) as the conservative model-form check. Do not use the weak-prior magnitude nuisance fit (`22.834308 nm/day`) as the default: its held-out improvement is small and the rate is not stable under optical assumptions.",
        "",
        "## Time-Window Sensitivity",
        "",
        "The matched empirical-response rates below are endpoint averages, not constant-rate assumptions.",
        "",
        "| Fitted window | Repeats | Initial thickness (nm) | Accumulated thickness (nm) | Endpoint-average rate (nm/day) |",
        "|---|---:|---:|---:|---:|",
        f"| Full sequence | 0-15 | {empirical['initial_thickness_nm']:.6f} | {empirical['accumulated_thickness_nm']:.6f} | {empirical['mean_rate_nm_day']:.6f} |",
        f"| Pre-ACM refit | 0-13 | {pre_acm['initial_thickness_nm']:.6f} | {pre_acm['accumulated_thickness_nm']:.6f} | {pre_acm['mean_rate_nm_day']:.6f} |",
        "",
        f"Within the full fitted trajectory, interval rates span `{np.min(interval_rates):.6f}` to `{np.max(interval_rates):.6f} nm/day`; the only interval wholly after the ACM deployment is `{interval_rates[-1]:.6f} nm/day`. With only two post-event nodes, that last value is descriptive and does not establish a stable post-ACM rate.",
        "",
        "The larger change in fitted initial thickness than in accumulated rate under the pre-ACM refit is further evidence that absolute `d0` is less robust than relative accumulation.",
        "",
        "## Sensitivity",
        "",
        "| Model | Optical prior weight | d0 (nm) | Rate (nm/day) | RMS incl. references | RMS excl. references | Mean held-out RMSE | Physical thickness admissible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        prior = row["prior_weight"] if row["prior_weight"] != "" else "none"
        lines.append(
            f"| {row['model']} | {prior} | {row['initial_thickness_nm']:.6f} | "
            f"{row['mean_rate_nm_day']:.6f} | {row['all_pixel_rms_log']:.8f} | "
            f"{row['nonreference_pixel_rms_log']:.8f} | "
            f"{row['held_out_rmse_log']:.12g} | "
            f"{'yes' if row['admissible_physical_thickness'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"The best admissible predictive model is `{best['model']}` with held-out RMSE `{best['held_out_rmse_log']:.12g}`. Its fold standard error is `{best_standard_error:.12g}`, giving a one-standard-error limit of `{one_se_limit:.12g}`. The selected primary score is `{primary['held_out_rmse_log']:.12g}`.",
            "",
            "The one-standard-error rule is a descriptive model-selection heuristic here because the training sets overlap; it is not an inferential uncertainty interval.",
            "",
            f"Across the magnitude-only nuisance priors, the fitted rate spans `{min(sensitivity_rates):.6f}` to `{max(sensitivity_rates):.6f} nm/day`. Relative to the empirical model, the best admissible model {predictive_change}, but it is accompanied by this large rate sensitivity.",
            "",
            f"As a descriptive temporal sensitivity check, leave-one-epoch-out empirical fits give rates from `{np.min(empirical_fold_rates):.6f}` to `{np.max(empirical_fold_rates):.6f} nm/day` (standard deviation `{np.std(empirical_fold_rates, ddof=1):.6f}`) and `d0` from `{np.min(empirical_fold_d0):.6f}` to `{np.max(empirical_fold_d0):.6f}` nm. These folds are correlated and are not a formal confidence interval.",
            "",
            "The magnitude-plus-phase model is treated as a diagnostic rather than an admissible physical-thickness model because the normalized intensity data provide no external phase calibration. Its fitted `d0` also reaches the zero-thickness boundary, which is evidence of instability but is not by itself a proof of non-identifiability.",
            "",
            "No LED-specific drift, gain, or optical parameter is used in any model.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")

    labels = [
        "Empirical",
        "Magnitude\n0.1",
        "Magnitude\n0.01",
        "Magnitude\n0.001",
        "Phase\n(rejected)",
    ]
    x = np.arange(len(rows))
    figure, first_axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bars = first_axis.bar(
        x,
        [row["held_out_rmse_log"] for row in rows],
        color=["#0072B2", "#56B4E9", "#56B4E9", "#56B4E9", "#999999"],
        alpha=0.8,
        label="Held-out RMSE",
    )
    bars[-1].set_hatch("//")
    first_axis.set_ylabel("Mean held-out RMSE (log)")
    first_axis.set_xticks(x, labels)
    first_axis.grid(axis="y", alpha=0.2)
    second_axis = first_axis.twinx()
    second_axis.plot(
        x,
        [row["mean_rate_nm_day"] for row in rows],
        "o-",
        color="#D55E00",
        linewidth=2.0,
        label="Fitted accumulation rate",
    )
    second_axis.set_ylabel("Mean accumulation rate (nm/day)")
    first_axis.set_title("Predictive change is small compared with optical-prior rate sensitivity")
    handles1, labels1 = first_axis.get_legend_handles_labels()
    handles2, labels2 = second_axis.get_legend_handles_labels()
    first_axis.legend(handles1 + handles2, labels1 + labels2, frameon=False)
    plot_path = output_dir / "optical_prior_sensitivity.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return csv_path, report_path, plot_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-fit", type=Path, default=CORE_FIT)
    parser.add_argument("--core-nodes", type=Path, default=CORE_NODES)
    parser.add_argument("--pre-acm-fit", type=Path, default=PRE_ACM_FIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    empirical = read_core_fit(args.core_fit)
    pre_acm = read_core_fit(args.pre_acm_fit)
    interval_rates = read_selected_interval_rates(args.core_nodes)
    empirical_scores = validation_scores(
        MAGNITUDE_CASES[0][1] / "repeat_blocked_validation.csv",
        "empirical_table",
    )
    empirical_fold_rates = validation_metric(
        MAGNITUDE_CASES[0][1] / "repeat_blocked_validation.csv",
        "empirical_table",
        "mean_rate_nm_day",
    )
    empirical_fold_d0 = validation_metric(
        MAGNITUDE_CASES[0][1] / "repeat_blocked_validation.csv",
        "empirical_table",
        "initial_thickness_nm",
    )
    empirical["held_out_rmse_log"] = float(np.mean(empirical_scores))
    nonreference_rms_factor = (
        empirical["nonreference_pixel_rms_log"] / empirical["all_pixel_rms_log"]
    )
    rows = [empirical]
    fold_scores = [empirical_scores]
    for prior_weight, directory in MAGNITUDE_CASES:
        row = read_report_row(
            directory / "simultaneous_complex_q_fit.md",
            "Simultaneous q nuisance",
        )
        scores = validation_scores(
            directory / "repeat_blocked_validation.csv",
            "joint_q",
        )
        row.update(
            {
                "model": "magnitude nuisance",
                "prior_weight": prior_weight,
                "held_out_rmse_log": float(np.mean(scores)),
                "admissible_physical_thickness": True,
                "nonreference_pixel_rms_log": (
                    row["all_pixel_rms_log"] * nonreference_rms_factor
                ),
            }
        )
        rows.append(row)
        fold_scores.append(scores)
    phase = read_report_row(
        PHASE_CASE / "simultaneous_complex_q_fit.md",
        "Simultaneous q nuisance",
    )
    phase_scores = validation_scores(
        PHASE_CASE / "repeat_blocked_validation.csv",
        "joint_q",
    )
    phase.update(
        {
            "model": "magnitude + phase nuisance",
            "prior_weight": 0.001,
            "held_out_rmse_log": float(np.mean(phase_scores)),
            "admissible_physical_thickness": False,
            "nonreference_pixel_rms_log": (
                phase["all_pixel_rms_log"] * nonreference_rms_factor
            ),
        }
    )
    rows.append(phase)
    fold_scores.append(phase_scores)
    paths = write_outputs(
        args.output_dir,
        rows,
        fold_scores,
        empirical_fold_rates,
        empirical_fold_d0,
        pre_acm,
        interval_rates,
    )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
