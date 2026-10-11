"""Plot CAR190 and CAR216 sinusoid residuals as detector tile envelopes."""

import argparse
from datetime import timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from analyze_wfi_post_reset_led_sinusoid import (
    DEFAULT_RESET_DAY,
    evaluate_model,
)
from plot_wfi_all_detector_temperature_sine_fit import (
    ACM_STEP_REPEAT,
    DETECTORS,
    acm_step_timestamp,
    envelope,
    fit_tile,
    parse_detectors,
    read_rows,
)


DEFAULT_LED = 3


def residual_value(result, row, reset_day):
    prediction = evaluate_model(
        np.asarray([row["days"]], dtype=float),
        result["fit"],
        reset_day,
    )[0]
    return 1000.0 * (row["value"] - prediction)


def plot_residuals(
    output_path,
    detector_results,
    led,
    reset_day,
    acm_step_utc,
):
    figure, axis = plt.subplots(figsize=(16, 10), constrained_layout=True)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(detector_results)))
    detector_handles = []
    for color, (detector, results) in zip(colors, detector_results.items()):
        train_utc, train_low, _, train_high = envelope(
            results,
            "train",
            lambda result, row: residual_value(result, row, reset_day),
        )
        selected_utc, selected_low, _, selected_high = envelope(
            results,
            "selected",
            lambda result, row: residual_value(result, row, reset_day),
        )
        axis.fill_between(
            train_utc,
            train_low,
            train_high,
            color=color,
            alpha=0.16,
        )
        axis.fill_between(
            selected_utc,
            selected_low,
            selected_high,
            color=color,
            alpha=0.48,
        )
        detector_handles.append(
            Patch(facecolor=color, alpha=0.48, label=detector)
        )

    axis.axvline(
        acm_step_utc,
        color="black",
        linestyle="--",
        linewidth=1.25,
    )
    axis.set_xlabel("Exposure start time (UTC)")
    axis.set_ylabel("Sinusoid residual (millilog)")
    axis.set_title(f"WFI LED {led}: detector tile residual envelopes, CAR190 and CAR216")
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc)
    )
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axis.grid(alpha=0.2)
    detector_legend = axis.legend(
        handles=detector_handles,
        title="detector color",
        frameon=False,
        ncol=3,
        loc="upper left",
        fontsize=9,
    )
    axis.add_artist(detector_legend)
    axis.legend(
        handles=[
            Patch(facecolor="0.35", alpha=0.16, label="CAR190 residual envelope"),
            Patch(facecolor="0.35", alpha=0.48, label="CAR216 residual envelope"),
            Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=1.25,
                label=(
                    f"ACM step: repeat {ACM_STEP_REPEAT}, "
                    f"{acm_step_utc.strftime('%Y-%m-%d %H:%M UTC')}"
                ),
            ),
        ],
        frameon=False,
        loc="lower left",
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument(
        "--detectors",
        type=parse_detectors,
        default=DETECTORS,
        help="Comma-separated detector list (default: all WFI detectors)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--led", type=int, default=DEFAULT_LED)
    parser.add_argument("--reset-day", type=float, default=DEFAULT_RESET_DAY)
    return parser.parse_args()


def main():
    args = parse_args()
    detector_results = {}
    all_rows = []
    for detector in args.detectors:
        statistics_path = args.statistics_dir / (
            f"{detector.lower()}_tile_log_statistics_512_count_corrected.csv"
        )
        rows = read_rows(statistics_path)
        all_rows.extend(rows)
        tiles = sorted({row["tile"] for row in rows})
        detector_results[detector] = [
            fit_tile(rows, tile, args.led, args.reset_day) for tile in tiles
        ]

    acm_step_utc = acm_step_timestamp(all_rows)
    plot_residuals(
        args.output,
        detector_results,
        args.led,
        args.reset_day,
        acm_step_utc,
    )
    tile_count = sum(len(results) for results in detector_results.values())
    print(f"Detectors: {', '.join(detector_results)}")
    print(f"Independent residual envelopes: {tile_count} tiles, LED {args.led}")
    print(
        "Tiles per detector: "
        + ", ".join(
            f"{detector}={len(results)}"
            for detector, results in detector_results.items()
        )
    )
    print("Fit model: CAR190 only")
    print("Displayed comparison envelope: CAR216")
    print(f"ACM step repeat {ACM_STEP_REPEAT}: {acm_step_utc.isoformat()}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
