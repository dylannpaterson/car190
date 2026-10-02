"""Render plots from an existing independent detector/LED count-space fit."""

import argparse
from pathlib import Path

import numpy as np

from fit_wfi_count_shift_scale_mcmc import load_model_data, save_plots


RESET_DAY = 15.064405023148149


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector", default="WFI02")
    parser.add_argument("--reset-day", type=float, default=RESET_DAY)
    parser.add_argument("--additional-acm-step-repeat", type=int, default=39)
    parser.add_argument("--pre-reset-acm-step-repeat", type=int, default=35)
    args = parser.parse_args()
    model = load_model_data(
        args.statistics,
        args.corrections,
        args.reset_day,
        args.additional_acm_step_repeat,
        args.pre_reset_acm_step_repeat,
        [args.detector],
    )
    with np.load(args.posterior, allow_pickle=False) as archive:
        chains = np.asarray(archive["chains"], dtype=float)
        log_probabilities = np.asarray(archive["log_probability"], dtype=float)
        labels = np.asarray(archive["detector_led"]).astype(str)
    detector = args.detector.upper()
    selected = [index for index, label in enumerate(labels) if label.startswith(f"{detector}_")]
    if len(selected) != len(model["segments"]):
        raise ValueError(
            f"Posterior contains {len(selected)} chains for {detector}, "
            f"but the loaded model has {len(model['segments'])} LED segments"
        )
    results = [
        {
            "chain": chains[index],
            "log_probability": log_probabilities[index],
        }
        for index in selected
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_plots(args.output_dir, model, results, args.detector)
    print(f"Wrote {args.detector}-only plots to {args.output_dir}")


if __name__ == "__main__":
    main()
