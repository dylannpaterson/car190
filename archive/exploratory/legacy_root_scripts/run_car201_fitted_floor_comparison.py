"""Repeat saved CAR201 fits with a detector-level fitted systematic floor."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PRODUCTION = ROOT / "20261106/corrected_statistics_4h_thinned/production"


def run_detector(detector):
    lower = detector.lower()
    experiments = PRODUCTION / lower / "spline_relative_response_normal/experiments"
    old = experiments / f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched"
    new = experiments / f"car201_zane_{lower}_ep3v2_ep4_faint_v2_tile_matched_fitted_floor"
    summary = json.loads((old / "car201_actual_star_modulation_fit_summary.json").read_text())
    new.mkdir(parents=True, exist_ok=True)
    output_summary = new / "car201_actual_star_modulation_fit_summary.json"
    if output_summary.is_file():
        existing = json.loads(output_summary.read_text())
        if existing["data_quality"]["systematic_floor_fitted"] and existing["fit"]["success"]:
            print(f"{detector}: using completed fitted-floor experiment", flush=True)
            return
    command = [
        sys.executable, str(ROOT / "fit_car201_zane_star_reflectance_modulation.py"),
        "--detector", detector,
        "--photometry-dir", str(ROOT.parent / "CAR201/zanes_photometry/20261106"),
        "--sed-posteriors", summary["sed_posteriors"],
        "--sed-grid", str(ROOT.parent / "CAR201/sed/v2/btsettl-spectral-grid.h5"),
        "--optical-cache-root", summary["ice_history"]["optical_state_source"],
        "--ice-history-fit-root", str(PRODUCTION / lower / "spline_relative_response_normal/fits"),
        "--output-dir", str(new),
        "--pass-set", "ep3v2_ep4", "--allow-partial-source-coverage",
        "--basis", summary["modulation"]["basis"],
        "--wavelength-stride-nm", str(summary["modulation"]["wavelength_stride_nm"]),
        "--maxent-weight", str(summary["modulation"]["maxent_regularization"]["weight"]),
        "--phase-smoothness-weight", str(summary["modulation"]["phase_smoothness_regularization"]["weight"]),
        "--fit-systematic-floor",
    ]
    if summary["repeat_photometry_outlier_screen"]["enabled_for_exclusion"]:
        command.append("--exclude-repeat-outliers")
    for exclusion in summary["manual_measurement_exclusions"]:
        command.extend(["--exclude-measurement", exclusion["source_id"], exclusion["filter"], exclusion["pass"]])
    (new / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    print(f"{detector}: fitting; log={new / 'fit.log'}", flush=True)
    with (new / "fit.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(output_summary.read_text())
    if not result["fit"]["success"]:
        raise RuntimeError(f"{detector}: fitted-floor optimizer did not converge")
    for key in ("matched_source_count", "active_source_count_with_usable_photometry"):
        if result[key] != summary[key]:
            raise RuntimeError(f"{detector}: changed source cohort ({key})")
    print(f"{detector}: complete; sigma_sys={result['data_quality']['systematic_floor_mmag']:.3f} mmag; "
          f"reduced chi2={result['fit']['reduced_chi2_approx']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="append", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(run_detector, args.detector))


if __name__ == "__main__":
    main()
