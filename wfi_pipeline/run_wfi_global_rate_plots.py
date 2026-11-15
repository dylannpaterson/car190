"""Build focal-plane and all-detector rate plots for a completed WFI run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "wfi_pipeline", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from wfi_pipeline.update_ice_history import _run_focal_plane_and_ice_trend_plots
from wfi_pipeline_defaults import DETECTORS, FULL_TMM, RELATIVE_RESPONSE


def _reset_layout(fit_roots):
    reference_days = None
    reference_names = None
    for detector in DETECTORS:
        archive_path = (
            Path(fit_roots[detector])
            / "fits/tile_000/posterior_samples.npz"
        )
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            if "reset_event_days" in archive:
                days = np.asarray(archive["reset_event_days"], dtype=float).reshape(-1)
                names = np.asarray(archive["reset_event_names"]).astype(str).reshape(-1)
            else:
                days = np.asarray(
                    [archive["ice_reset_day"]]
                    + ([archive["third_decon_day"]]
                       if "third_decon_day" in archive else []),
                    dtype=float,
                ).reshape(-1)
                names = np.asarray(
                    ["second_decon"]
                    + (["third_decon"] if days.size > 1 else [])
                ).astype(str)
        if days.size == 0 or names.size != days.size or np.any(np.diff(days) <= 0.0):
            raise ValueError(f"{detector}: invalid reset-event metadata in {archive_path}")
        if reference_days is None:
            reference_days = days
            reference_names = names
        elif not np.allclose(days, reference_days, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{detector}: reset-event days differ from other detectors")
        elif not np.array_equal(names, reference_names):
            raise ValueError(f"{detector}: reset-event names differ from other detectors")
    return reference_days


def build_global_rate_plots(statistics_dir, production_root, route, output_dir):
    route = str(route)
    if route not in (RELATIVE_RESPONSE, FULL_TMM):
        raise ValueError(f"Unknown model route: {route}")
    fit_subdir = "relative_response" if route == RELATIVE_RESPONSE else "full_tmm"
    production_root = Path(production_root).expanduser().resolve()
    fit_roots = {
        detector: production_root / detector.lower() / fit_subdir
        for detector in DETECTORS
    }
    reset_days = _reset_layout(fit_roots)
    outputs = _run_focal_plane_and_ice_trend_plots(
        statistics_dir,
        output_dir,
        fit_roots,
        reset_day=float(reset_days[0]),
        third_decon_day=(float(reset_days[1]) if reset_days.size > 1 else None),
        reset_days=reset_days,
    )
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--route", choices=(RELATIVE_RESPONSE, FULL_TMM), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    outputs = build_global_rate_plots(
        args.statistics_dir,
        args.production_root,
        args.route,
        args.output_dir,
    )
    print(json.dumps({"plots": outputs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
