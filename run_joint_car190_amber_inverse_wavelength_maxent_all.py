"""Run explicit detector selections sequentially for the raw-data joint model."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import run_amber_student_t_detectors as run


ROOT = Path(__file__).resolve().parent
FIT_SCRIPT = ROOT / "run_joint_car190_amber_inverse_wavelength_maxent.py"
DEFAULT_STATISTICS_DIR = ROOT / "20261111/corrected_statistics_4h_thinned"
EXPERIMENT = "car190_amber_joint_2day_reciprocal_maxent"


def run_sequential(detectors, statistics_dir, maxiter, ftol, timeout_minutes):
    records = []
    for index, detector in enumerate(detectors, start=1):
        production = Path(statistics_dir) / "production" / detector.lower()
        relative_root = production / "relative_response"
        fit_dir = relative_root / "experiments" / EXPERIMENT
        log_root = relative_root / "experiments" / f"{EXPERIMENT}_run_logs"
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{detector.lower()}.log"
        summary_path = fit_dir / "summary.json"
        checkpoint_path = fit_dir / "optimizer_checkpoint.npz"

        if summary_path.is_file():
            saved = json.loads(summary_path.read_text())
            if saved.get("optimizer_success"):
                record = {
                    "detector": detector,
                    "status": "already_complete",
                    "fit_directory": str(fit_dir),
                    "log": str(log_path),
                }
                records.append(record)
                print(f"{detector}: already complete; skipping", flush=True)
                continue

        command = [
            sys.executable,
            str(FIT_SCRIPT),
            "--detector",
            detector,
            "--statistics-dir",
            str(statistics_dir),
            "--maxiter",
            str(maxiter),
            "--ftol",
            str(ftol),
        ]
        if checkpoint_path.is_file():
            command.append("--resume")
        elif fit_dir.exists() and any(fit_dir.iterdir()):
            record = {
                "detector": detector,
                "status": "blocked_existing_output_without_checkpoint",
                "fit_directory": str(fit_dir),
                "log": str(log_path),
            }
            records.append(record)
            print(f"{detector}: existing incomplete output has no checkpoint", flush=True)
            continue

        print(
            f"=== {detector} ({index}/{len(detectors)}): no warm-up; "
            f"maxiter={maxiter}; time limit={timeout_minutes} min ===",
            flush=True,
        )
        started = time.monotonic()
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nCOMMAND: {' '.join(command)}\n")
                log.flush()
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_minutes * 60.0,
                    check=False,
                )
            status = "completed" if result.returncode == 0 else f"failed_exit_{result.returncode}"
            record = {
                "detector": detector,
                "status": status,
                "return_code": int(result.returncode),
                "seconds": time.monotonic() - started,
                "fit_directory": str(fit_dir),
                "log": str(log_path),
            }
        except subprocess.TimeoutExpired:
            record = {
                "detector": detector,
                "status": "time_limit_checkpoint_saved",
                "seconds": time.monotonic() - started,
                "fit_directory": str(fit_dir),
                "log": str(log_path),
                "checkpoint_exists": checkpoint_path.is_file(),
            }
        records.append(record)
        print(json.dumps(record), flush=True)

    report_path = Path(statistics_dir) / "production" / "wavelength_response_bundle" / (
        "joint_car190_amber_inverse_wavelength_maxent_run_status.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {}
    if report_path.is_file():
        saved = json.loads(report_path.read_text())
        combined.update({row["detector"]: row for row in saved.get("detectors", [])})
    combined.update({row["detector"]: row for row in records})
    ordered = [combined[name] for name in run.workflow.DETECTORS if name in combined]
    report_path.write_text(json.dumps({"detectors": ordered}, indent=2) + "\n")
    print(f"Wrote sequential run status to {report_path}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="append", required=True, choices=run.workflow.DETECTORS,
                        help="One explicit detector; repeated values run sequentially")
    parser.add_argument("--statistics-dir", type=Path, default=DEFAULT_STATISTICS_DIR)
    parser.add_argument("--maxiter", type=int, default=700)
    parser.add_argument("--ftol", type=float, default=1e-7)
    parser.add_argument("--timeout-minutes", type=int, default=60)
    args = parser.parse_args()
    detectors = list(dict.fromkeys(args.detector))
    if args.maxiter <= 0 or not math.isfinite(args.ftol) or args.ftol <= 0.0 \
            or args.timeout_minutes <= 0:
        parser.error("--maxiter, --ftol, and --timeout-minutes must be positive")
    run_sequential(detectors, args.statistics_dir, args.maxiter, args.ftol, args.timeout_minutes)


if __name__ == "__main__":
    main()
