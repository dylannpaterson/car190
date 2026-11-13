"""Run detector-specific fixed-ice and 2-day joint CAR190/Amber experiments.

Pass each detector explicitly with a repeated ``--detector`` option. Fits run
sequentially; checkpoints, summaries, and per-stage logs stay under the
detector's production experiment tree.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

import run_amber_student_t_detectors as run
from experiment_joint_wfi01_fixed_ice import fixed_output
from experiment_joint_wfi01_weekly_ice import cadence_output


ROOT = Path(__file__).resolve().parent
FIT_SCRIPT = ROOT / "experiment_joint_wfi01_weekly_ice.py"
FIXED_SCRIPT = ROOT / "experiment_joint_wfi01_fixed_ice.py"
SUMMARY_SCRIPT = ROOT / "summarize_joint_wfi01_weekly_ice.py"
MANIFEST = ROOT / "joint_car190_amber_2day_all_detectors_manifest.json"
CSV_SUMMARY = ROOT / "joint_car190_amber_2day_all_detectors_summary.csv"


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def save_manifest(results):
    payload = {
        "run_type": "per-detector joint CAR190 + Amber fit with 2-day per-tile ice-rate controls",
        "detector_selection_policy": "each WFI detector passed explicitly; sequential execution",
        "results": results,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    rows = []
    for detector in run.workflow.DETECTORS:
        if detector in results:
            rows.append({"detector": detector, **results[detector]})
    pd.DataFrame(rows).to_csv(CSV_SUMMARY, index=False)


def run_logged(detector, command, log_path, stage):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print(f"{detector}: starting {stage}", flush=True)
    with log_path.open("a") as log:
        log.write(f"\n[{started}] COMMAND: {' '.join(map(str, command))}\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    print(f"{detector}: {stage} process exited {result.returncode}; log={log_path}", flush=True)
    return result.returncode


def collect_result(detector, fixed_summary, fit_summary, comparison):
    metrics_path = cadence_output(detector, 2.0) / "fixed_vs_ice_stellar_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    metric_rows = {row["model"]: row for _, row in metrics.iterrows()}
    baseline = metric_rows.get("fixed ice", {})
    fitted = metric_rows.get("2-day per-tile ice", {})
    return {
        "status": "complete",
        "fixed_ice_optimizer_success": bool(fixed_summary.get("optimizer_success", False)),
        "two_day_optimizer_success": bool(fit_summary.get("optimizer_success", False)),
        "two_day_optimizer_message": fit_summary.get("optimizer_message"),
        "two_day_total_ice_parameters": fit_summary.get("total_ice_parameters"),
        "two_day_ice_parameter_count_per_tile": fit_summary.get("ice_parameters_per_tile"),
        "amber_stars": fit_summary.get("stellar_stars"),
        "amber_measurements": fit_summary.get("stellar_measurements"),
        "fixed_car190_chi2": fixed_summary.get("final_LED_chi2"),
        "fixed_amber_residual_rms_mmag": baseline.get("residual_rms_mmag"),
        "two_day_amber_residual_rms_mmag": fitted.get("residual_rms_mmag"),
        "two_day_amber_fraction_within_0p5_percent": fitted.get("fraction_within_0p5_percent"),
        "two_day_prediction_change_rms_mmag": comparison.get("prediction_change_rms_mmag"),
        "two_day_max_abs_ice_change_nm": fit_summary.get("max_abs_ice_change_nm"),
        "fixed_fit_directory": str(fixed_output(detector)),
        "two_day_fit_directory": str(cadence_output(detector, 2.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="append", required=True, choices=run.workflow.DETECTORS)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--max-resume-passes", type=int, default=5)
    args = parser.parse_args()
    detectors = list(dict.fromkeys(args.detector))

    existing = read_json(MANIFEST) or {"results": {}}
    results = existing.get("results", {})
    for detector in detectors:
        print(f"\n=== {detector}: detector fit {detectors.index(detector) + 1}/{len(detectors)} ===", flush=True)
        fixed_dir = fixed_output(detector)
        fixed_summary_path = fixed_dir / "joint_fit_summary.json"
        if not (read_json(fixed_summary_path) or {}).get("optimizer_success", False):
            code = run_logged(
                detector,
                [sys.executable, str(FIXED_SCRIPT), "--detector", detector],
                fixed_dir / "fixed_ice_joint_fit.log",
                "fixed-ice joint fit",
            )
            fixed_summary = read_json(fixed_summary_path)
            if code != 0 or not (fixed_summary or {}).get("optimizer_success", False):
                results[detector] = {"status": "fixed_ice_fit_failed", "exit_code": code,
                                     "fixed_fit_directory": str(fixed_dir)}
                save_manifest(results)
                continue
        fixed_summary = read_json(fixed_summary_path) or {}

        fit_dir = cadence_output(detector, 2.0)
        fit_summary_path = fit_dir / "summary.json"
        for attempt in range(args.max_resume_passes + 1):
            fit_summary = read_json(fit_summary_path)
            if (fit_summary or {}).get("optimizer_success", False):
                break
            command = [sys.executable, str(FIT_SCRIPT), "--detector", detector,
                       "--spacing-days", "2", "--maxiter", str(args.maxiter)]
            checkpoint = fit_dir / "optimizer_checkpoint.npz"
            if checkpoint.is_file():
                command.append("--resume")
            elif (cadence_output(detector, 7.0) / "joint_weekly_ice_fit.npz").is_file():
                command.append("--initialize-from-weekly")
            code = run_logged(detector, command, fit_dir / "two_day_joint_fit.log",
                              f"2-day fit pass {attempt + 1}")
            fit_summary = read_json(fit_summary_path)
            if (fit_summary or {}).get("optimizer_success", False):
                break
            if not checkpoint.is_file() or attempt == args.max_resume_passes:
                results[detector] = {"status": "two_day_fit_failed", "exit_code": code,
                                     "optimizer_message": (fit_summary or {}).get("optimizer_message"),
                                     "two_day_fit_directory": str(fit_dir)}
                save_manifest(results)
                break
        else:
            fit_summary = read_json(fit_summary_path)

        fit_summary = read_json(fit_summary_path) or {}
        if not fit_summary.get("optimizer_success", False):
            continue

        comparison_path = fit_dir / "comparison_summary.json"
        if not comparison_path.is_file():
            run_logged(
                detector,
                [sys.executable, str(SUMMARY_SCRIPT), "--detector", detector, "--fit-dir", str(fit_dir)],
                fit_dir / "diagnostics.log",
                "fixed-vs-2-day diagnostics",
            )
        comparison = read_json(comparison_path)
        if comparison is None:
            results[detector] = {"status": "diagnostics_failed", "two_day_fit_directory": str(fit_dir)}
        else:
            results[detector] = collect_result(detector, fixed_summary, fit_summary, comparison)
        save_manifest(results)
        print(f"{detector}: {results[detector]['status']}", flush=True)

    save_manifest(results)
    print(f"\nManifest: {MANIFEST}\nSummary CSV: {CSV_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
