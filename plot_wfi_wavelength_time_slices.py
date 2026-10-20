"""Plot wavelength-response time slices in the style of the Eric comparison."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wfi_relative_response import evaluate_spectral_response_cache, load_spectral_response_cache


DEFAULT_TIMES = (
    "2026-09-12T02:00:00",
    "2026-09-19T16:00:00",
    "2026-09-28T22:00:00",
    "2026-10-06T13:00:00",
    "2026-10-14T14:00:00",
)
DEFAULT_WAVELENGTHS = np.arange(480.0, 2300.1, 2.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-cache", type=Path, required=True)
    parser.add_argument("--fit-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", default="WFI16")
    parser.add_argument("--tile-id", type=int, default=136)
    parser.add_argument("--time", dest="times", action="append")
    return parser.parse_args()


def load_observation_times(fit):
    statistics_path = Path(str(np.asarray(fit["statistics_path"]).reshape(-1)[0]))
    with statistics_path.open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["group"]))
    return np.asarray([row["exposure_start_time"] for row in rows])


def nearest_observation(times, target):
    values = np.asarray([np.datetime64(str(value)[:26]) for value in times])
    target_value = np.datetime64(target)
    distance = np.abs(values.astype("datetime64[s]") - target_value.astype("datetime64[s]"))
    return int(np.argmin(distance.astype("timedelta64[s]").astype(np.int64)))


def main():
    args = parse_args()
    target_times = tuple(args.times or DEFAULT_TIMES)
    cache = load_spectral_response_cache(
        args.spectral_cache,
        detector=args.detector,
        tile_id=args.tile_id,
    )
    with np.load(args.fit_archive, allow_pickle=False) as source:
        fit = {name: np.asarray(source[name]) for name in source.files}
    observation_times = load_observation_times(fit)
    observation_days = np.asarray(fit["observation_days"], dtype=float).reshape(-1)
    thickness = np.asarray(fit["tile_ice_thickness"], dtype=float).reshape(-1)
    grey_offset = float(np.asarray(fit["post_reset_tile_log_offset"]).reshape(-1)[0])
    reset_day = float(np.asarray(fit["ice_reset_day"]).reshape(-1)[0])
    wavelengths = DEFAULT_WAVELENGTHS
    selected = []
    for target in target_times:
        index = nearest_observation(observation_times, target)
        selected.append((target, index))

    thickness_queries = np.asarray([thickness[index] for _, index in selected])
    _, response = evaluate_spectral_response_cache(
        cache,
        thickness_queries,
        wavelengths,
    )
    response = np.asarray(response, dtype=float)
    grey = np.asarray(
        [grey_offset if observation_days[index] >= reset_day else 0.0 for _, index in selected]
    )
    response_percent = 100.0 * np.expm1(response + grey[:, None])

    figure, axis = plt.subplots(figsize=(14.0, 6.5), constrained_layout=True)
    colors = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(selected)))
    for selected_index, (color, (target, index), values) in enumerate(
        zip(colors, selected, response_percent)
    ):
        axis.plot(
            wavelengths,
            values,
            color=color,
            linewidth=2.0,
            label=f"{target[:16].replace('T', ' ')} UTC",
        )
        print(
            f"{target}: matched {observation_times[index]}, "
            f"day={observation_days[index]:.6f}, ice={thickness[index]:.6f} nm, "
            f"grey_log={grey[selected_index]:.9g}",
            flush=True,
        )
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Relative response - 1 (%)")
    axis.set_title(
        f"{args.detector}, tile {args.tile_id:03d} "
        "(pixel 2048, 2048): relative response against wavelength\n"
        "early to late; fitted ice trajectory plus post-reset grey throughput"
    )
    axis.grid(True, color="0.88")
    axis.legend(loc="upper right", frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
