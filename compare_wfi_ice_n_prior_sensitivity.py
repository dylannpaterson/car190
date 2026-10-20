"""Compare strict production tile fits under two constant ice-n priors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


RESET_DAY = 12.907397928240743
ICE_INDEX_REFERENCE = 1.25
CASES = {
    "WFI01": ("wfi01_baseline", "wfi01_reference_1p29_sigma_0p03"),
    "WFI10": ("wfi10_baseline", "wfi10_reference_1p29_sigma_0p03"),
}


def _fit_dir(root, name):
    return Path(root) / name / "fits" / "tile_136"


def _load_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _stats_weights(path):
    rows = _load_rows(path)
    return {int(row["group"]): float(row["n_pair_valid"]) for row in rows}


def _residual_metrics(fit_dir):
    rows = _load_rows(fit_dir / "tile_group_predictions.csv")
    weights_by_group = _stats_weights(next(fit_dir.parent.parent.glob("tile_statistics/*.csv")))
    residual = np.asarray([float(row["residual_log"]) for row in rows])
    weights = np.asarray([weights_by_group[int(row["group"])] for row in rows])
    days = np.asarray([float(row["days"]) for row in rows])
    metrics = {}
    for label, mask in (
        ("all", np.ones(days.shape, dtype=bool)),
        ("pre", days < RESET_DAY),
        ("post", days >= RESET_DAY),
    ):
        metrics[f"{label}_residual_mean_log"] = float(np.mean(residual[mask]))
        metrics[f"{label}_residual_rms_log"] = float(np.sqrt(np.mean(residual[mask] ** 2)))
        metrics[f"{label}_count_weighted_rms_log"] = float(
            np.sqrt(np.sum(weights[mask] * residual[mask] ** 2) / np.sum(weights[mask]))
        )
    return rows, residual, metrics


def _rate_metrics(fit_dir):
    with np.load(fit_dir / "posterior_samples.npz", allow_pickle=False) as archive:
        rates = np.asarray(archive["instantaneous_rate_nm_day"], dtype=float).reshape(-1)
        days = np.asarray(archive["observation_days"], dtype=float).reshape(-1)
        ice_n = ICE_INDEX_REFERENCE * np.exp(
            float(np.asarray(archive["ice_n_log_constant"]).reshape(-1)[0])
        )
    pre = days < RESET_DAY
    post = ~pre
    with (fit_dir / "constant_rate_map.json").open() as handle:
        constant_rate = float(json.load(handle)["rate_nm_day"])
    return {
        "constant_rate_scale_nm_day": constant_rate,
        "pre_rate_nm_day": float(np.mean(rates[pre])),
        "post_rate_nm_day": float(np.mean(rates[post])),
        "pre_minus_post_rate_nm_day": float(np.mean(rates[pre]) - np.mean(rates[post])),
        "ice_n_fitted": ice_n,
    }


def compare(root, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    led_rows = []
    for detector, (baseline_name, modified_name) in CASES.items():
        baseline_dir = _fit_dir(root, baseline_name)
        modified_dir = _fit_dir(root, modified_name)
        baseline_group_rows, baseline_residual, baseline_residual_metrics = _residual_metrics(
            baseline_dir
        )
        modified_group_rows, modified_residual, modified_residual_metrics = _residual_metrics(
            modified_dir
        )
        if [row["group"] for row in baseline_group_rows] != [row["group"] for row in modified_group_rows]:
            raise ValueError(f"Group ordering differs for {detector}")
        rate_baseline = _rate_metrics(baseline_dir)
        rate_modified = _rate_metrics(modified_dir)
        baseline_weight = np.asarray(
            [_stats_weights(next(baseline_dir.parent.parent.glob("tile_statistics/*.csv")))[int(row["group"])] for row in baseline_group_rows]
        )
        residual_delta = modified_residual - baseline_residual
        summary_rows.append(
            {
                "detector": detector,
                "prior": "baseline_1p25_sigma_0p01",
                **rate_baseline,
                **baseline_residual_metrics,
                "residual_change_rms_log": 0.0,
                "residual_change_max_abs_log": 0.0,
                "baseline_modified_residual_correlation": 1.0,
            }
        )
        summary_rows.append(
            {
                "detector": detector,
                "prior": "reference_1p29_sigma_0p03",
                **rate_modified,
                **modified_residual_metrics,
                "residual_change_rms_log": float(np.sqrt(np.mean(residual_delta**2))),
                "residual_change_max_abs_log": float(np.max(np.abs(residual_delta))),
                "baseline_modified_residual_correlation": float(
                    np.corrcoef(baseline_residual, modified_residual)[0, 1]
                ),
            }
        )
        for label, name, rows, residual in (
            ("baseline", baseline_name, baseline_group_rows, baseline_residual),
            ("reference_1p29_sigma_0p03", modified_name, modified_group_rows, modified_residual),
        ):
            for side, mask in (
                ("pre", np.asarray([float(row["days"]) < RESET_DAY for row in rows])),
                ("post", np.asarray([float(row["days"]) >= RESET_DAY for row in rows])),
            ):
                for led in sorted({int(row["led"]) for row in rows}):
                    led_mask = mask & np.asarray([int(row["led"]) == led for row in rows])
                    led_residual = residual[led_mask]
                    led_rows.append(
                        {
                            "detector": detector,
                            "prior": label,
                            "side": side,
                            "led": led,
                            "residual_mean_log": float(np.mean(led_residual)),
                            "residual_rms_log": float(np.sqrt(np.mean(led_residual**2))),
                            "count": int(led_residual.size),
                        }
                    )
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "residual_by_led.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=led_rows[0].keys())
        writer.writeheader()
        writer.writerows(led_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.root, args.output_dir)


if __name__ == "__main__":
    main()
