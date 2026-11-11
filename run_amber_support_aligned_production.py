"""Run support-aligned, free-endpoint Amber fits sequentially by detector."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

import run_amber_student_t_detectors as run
from experiment_free_led_modulation import expand_parameters
from experiment_support_aligned_endpoints import solve
from support_aligned_modulation import effective_area_support, configure_basis


def directory(detector):
    _, old = run.canonical_configuration(detector)
    return old.with_name(old.name + "_support_aligned_free_endpoints_bound0p60")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="append", required=True, choices=run.workflow.DETECTORS)
    args = parser.parse_args()
    for detector in args.detector:
        area, old = run.canonical_configuration(detector)
        output = directory(detector)
        finished = output / "fit_summary.json"
        if finished.exists() and json.loads(finished.read_text()).get("optimizer_success"):
            print(f"{detector}: completed fit already exists", flush=True)
            continue
        old_summary = json.loads((old / "fit_summary.json").read_text())
        with np.load(old / "predictions.npz", allow_pickle=False) as a:
            ids_old, valid_old, theta = a["source_ids"].astype(str), a["valid"], a["student_t_fit_parameters"]
        _, common, ids, mag, err, valid, frozen, model, _ = run.prepare(
            detector, write_inputs=False, effective_area_path=area, output_dir=output,
            full_effective_area_support=True)
        np.testing.assert_array_equal(ids_old, ids)
        np.testing.assert_array_equal(valid_old, valid)
        boundaries = effective_area_support(area)
        configure_basis(model, boundaries, free_endpoints=False)
        anchored = solve(model, mag, err, valid, theta, old_summary["extra_student_t_scale_mmag"],
                         None, output / "support_aligned_anchored_initializer")
        configure_basis(model, boundaries, free_endpoints=True)
        initial = expand_parameters(anchored["theta"], 7, 2)
        final = solve(model, mag, err, valid, initial, anchored["summary"]["extra_student_t_scale_mmag"],
                      .6, output)
        summary = final["summary"]
        summary.update({"detector": detector, "photometry": "Amber", "likelihood": "Student-t", "nu": 4.,
                        "used_stars": len(ids), "used_observations": int(valid.sum()),
                        "effective_area_file": str(area), "effective_area_sha256": hashlib.sha256(area.read_bytes()).hexdigest(),
                        "endpoint_basis": "detector effective-area support; free endpoints, fixed six LED anchors",
                        "selection": "same common Amber/Zane quality-selected repeated measurements as prior canonical fit",
                        "ice_histories_refitted": False})
        finished.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{detector}: complete; scale={summary['extra_student_t_scale_mmag']:.4f} mmag; bounds={summary['parameters_at_bounds']}", flush=True)


if __name__ == "__main__":
    main()
