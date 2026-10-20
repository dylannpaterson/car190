"""Validate small per-LED model offsets on chronological held-out repeats."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional diagnostic output
    plt = None
import numpy as np

import fit_wfi_tmm_production as production


DETECTORS = ("wfi01", "wfi10")
VARIANT_NAMES = (
    ("baseline", "Production baseline"),
    ("mod002", "AR/HgCdTe modulation, sigma=0.02"),
    ("mod010", "AR/HgCdTe modulation, sigma=0.10"),
)
EFFECTIVE_LED_WAVELENGTHS = np.asarray(
    [
        622.344620431963,
        846.342064885868,
        1031.7737605274665,
        1261.2256752723229,
        1482.4771405873898,
        1658.734259886027,
    ],
    dtype=float,
)


TAU_LOG_DEFAULT = 5.0e-4


def load_predictions(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["group"]))
    return {
        "group": np.asarray([int(row["group"]) for row in rows]),
        "days": np.asarray([float(row["days"]) for row in rows]),
        "led": np.asarray([int(row["led"]) for row in rows]),
        "repeat": np.asarray([int(row["repeat"]) for row in rows]),
        "residual": np.asarray([float(row["residual_log"]) for row in rows]),
    }


def variant_path(root, detector, variant):
    if variant == "baseline":
        return (
            root
            / detector
            / "n_prior_sigma_0p05"
            / "fits"
            / "tile_136"
            / "tile_group_predictions.csv"
        )
    suffix = "" if variant == "mod002" else "_modsigma010"
    return (
        root
        / detector
        / "diagnostics"
        / f"tile_136_hgcdte_modulation_matched_np_nprior005{suffix}"
        / "tile_group_predictions.csv"
    )


def load_counts(path):
    with Path(path).open(newline="") as handle:
        return {
            int(row["group"]): float(row["n_pair_valid"])
            for row in csv.DictReader(handle)
        }


def load_variant(root, detector, variant):
    values = load_predictions(variant_path(root, detector, variant))
    statistics_path = (
        root
        / detector
        / "n_prior_sigma_0p05"
        / "tile_statistics"
        / f"{detector}_tile_log_statistics_256_tile_136.csv"
    )
    counts_by_group = load_counts(statistics_path)
    try:
        values["counts"] = np.asarray(
            [counts_by_group[int(group)] for group in values["group"]],
            dtype=float,
        )
    except KeyError as error:
        raise ValueError(f"Missing count for prediction group {error.args[0]}") from error
    return values


def repeat_order(values):
    repeat_days = {}
    for repeat in np.unique(values["repeat"]):
        if repeat != 0:
            repeat_days[int(repeat)] = float(
                np.mean(values["days"][values["repeat"] == repeat])
            )
    return np.asarray(
        sorted(repeat_days, key=repeat_days.__getitem__),
        dtype=int,
    )


def chronological_splits(values):
    repeats = repeat_order(values)
    split = len(repeats) // 2
    if split < 1 or split == len(repeats):
        raise ValueError("Need at least two non-reference repeat nodes")
    early = repeats[:split]
    late = repeats[split:]
    return {
        "early_to_late": (early, late),
        "late_to_early": (late, early),
    }


def fit_led_offsets(values, train_repeats, tau_log):
    train = (values["repeat"] != 0) & np.isin(values["repeat"], train_repeats)
    likelihood_precision = (
        values["counts"]
        / np.median(values["counts"])
        / production.LOG_NOISE_SIGMA**2
    )
    offsets = np.zeros(6, dtype=float)
    prior_precision = 1.0 / tau_log**2
    for led in range(1, 7):
        selected = train & (values["led"] == led)
        precision = likelihood_precision[selected]
        offsets[led - 1] = float(
            np.sum(precision * values["residual"][selected])
            / (np.sum(precision) + prior_precision)
        )
    return offsets


def metrics(values, selected, offsets=None, effective_wavelengths=None):
    residual = values["residual"][selected].copy()
    if offsets is not None:
        residual -= offsets[values["led"][selected] - 1]
    counts = values["counts"][selected]
    wavelengths = None
    if effective_wavelengths is not None:
        wavelengths = effective_wavelengths[values["led"][selected] - 1]
    led_means = np.asarray(
        [
            np.mean(residual[values["led"][selected] == led])
            for led in range(1, 7)
        ]
    )
    return {
        "rms_log": float(np.sqrt(np.mean(residual**2))),
        "count_weighted_rms_log": float(
            np.sqrt(np.sum(counts * residual**2) / np.sum(counts))
        ),
        "residual_wavelength_pearson": float(
            np.corrcoef(residual, wavelengths)[0, 1]
        )
        if wavelengths is not None
        else float("nan"),
        "led_mean_wavelength_pearson": float(
            np.corrcoef(led_means, effective_wavelengths)[0, 1]
        )
        if effective_wavelengths is not None
        else float("nan"),
    }


def evaluate(root, tau_log):
    effective_wavelengths = EFFECTIVE_LED_WAVELENGTHS
    results = []
    for detector in DETECTORS:
        for variant, _ in VARIANT_NAMES:
            values = load_variant(root, detector, variant)
            for fold, (train_repeats, test_repeats) in chronological_splits(values).items():
                train = (values["repeat"] != 0) & np.isin(
                    values["repeat"], train_repeats
                )
                test = (values["repeat"] != 0) & np.isin(
                    values["repeat"], test_repeats
                )
                offsets = fit_led_offsets(values, train_repeats, tau_log)
                train_raw = metrics(values, train, effective_wavelengths=effective_wavelengths)
                train_adjusted = metrics(
                    values,
                    train,
                    offsets=offsets,
                    effective_wavelengths=effective_wavelengths,
                )
                test_raw = metrics(values, test, effective_wavelengths=effective_wavelengths)
                test_adjusted = metrics(
                    values,
                    test,
                    offsets=offsets,
                    effective_wavelengths=effective_wavelengths,
                )
                row = {
                    "detector": detector.upper(),
                    "variant": variant,
                    "fold": fold,
                    "tau_log": tau_log,
                    "train_repeat_count": len(train_repeats),
                    "test_repeat_count": len(test_repeats),
                    "train_days_min": float(np.min(values["days"][train])),
                    "train_days_max": float(np.max(values["days"][train])),
                    "test_days_min": float(np.min(values["days"][test])),
                    "test_days_max": float(np.max(values["days"][test])),
                }
                for prefix, result in (
                    ("train_raw", train_raw),
                    ("train_adjusted", train_adjusted),
                    ("test_raw", test_raw),
                    ("test_adjusted", test_adjusted),
                ):
                    row.update(
                        {
                            f"{prefix}_{name}": value
                            for name, value in result.items()
                        }
                    )
                row.update(
                    {
                        f"offset_led_{led}": float(offsets[led - 1])
                        for led in range(1, 7)
                    }
                )
                results.append(row)
    return results


def write_csv(results, path):
    fields = list(results[0])
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def plot_results(results, path):
    if plt is None:
        return False
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    variants = [variant for variant, _ in VARIANT_NAMES]
    labels = [variant.upper() for variant in variants]
    x = np.arange(len(variants))
    width = 0.36
    for detector_index, detector in enumerate(DETECTORS):
        for fold_index, fold in enumerate(("early_to_late", "late_to_early")):
            axis = axes[detector_index, fold_index]
            selected = [
                row
                for row in results
                if row["detector"] == detector.upper() and row["fold"] == fold
            ]
            selected.sort(key=lambda row: variants.index(row["variant"]))
            raw = np.asarray([row["test_raw_rms_log"] for row in selected]) * 1000.0
            adjusted = (
                np.asarray([row["test_adjusted_rms_log"] for row in selected])
                * 1000.0
            )
            axis.bar(x - width / 2.0, raw, width, label="raw")
            axis.bar(x + width / 2.0, adjusted, width, label="LED offset")
            axis.set_xticks(x, labels)
            axis.set_ylabel("Held-out RMS [10^-3]")
            axis.set_title(f"{detector.upper()} {fold.replace('_', ' -> ')}")
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("WFI tile 136: chronological held-out LED-offset check", y=1.02)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "20261012/corrected_statistics_program1063_4h/production/tile_size_256"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau-log", type=float, default=TAU_LOG_DEFAULT)
    return parser.parse_args()


def main():
    args = parse_args()
    if not np.isfinite(args.tau_log) or args.tau_log <= 0.0:
        raise ValueError("--tau-log must be positive and finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate(args.root, args.tau_log)
    write_csv(results, args.output_dir / "tile136_led_offset_holdout_validation.csv")
    plot_path = args.output_dir / "tile136_led_offset_holdout_validation.png"
    if not plot_results(results, plot_path):
        print("matplotlib unavailable; wrote CSV validation only")
    for path in sorted(args.output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
