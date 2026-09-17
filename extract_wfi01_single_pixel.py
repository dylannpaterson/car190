"""Extract one WFI01 pixel across all LED epochs for the shared fit."""

import argparse
import csv
from pathlib import Path

import asdf
import numpy as np

from analyze_wfi01_all_pixel_soc_model import DATA_DIR, INDEX_PATH, read_wfi01_index


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row", type=int, default=2044)
    parser.add_argument("--column", type=int, default=2044)
    return parser.parse_args()


def read_pixel(path, row, column):
    with asdf.open(path, lazy_load=True, ignore_unrecognized_tag=True) as af:
        model = af.tree["roman"]
        data = model["data"]
        dq = model["dq"]
        if row < 0 or column < 0 or row >= data.shape[0] or column >= data.shape[1]:
            raise IndexError(f"Pixel ({row}, {column}) is outside image shape {data.shape}")
        return float(data[row, column]), int(dq[row, column])


def extract_pixel(observations, row, column):
    values = []
    for observation in observations:
        value, dq = read_pixel(observation["image_path"], row, column)
        values.append({**observation, "value": value, "dq": dq})

    for led in range(1, 7):
        led_values = [item for item in values if item["led"] == led]
        baseline = led_values[0]
        if baseline["dq"] != 0 or not np.isfinite(baseline["value"]) or baseline["value"] <= 0.0:
            raise ValueError(
                f"Pixel ({row}, {column}) has invalid LED {led} baseline: "
                f"value={baseline['value']}, dq={baseline['dq']}"
            )
        for item in led_values:
            if item["dq"] != 0 or not np.isfinite(item["value"]) or item["value"] <= 0.0:
                raise ValueError(
                    f"Pixel ({row}, {column}) has invalid LED {led}, repeat "
                    f"{item['repeat']}: value={item['value']}, dq={item['dq']}"
                )
            item["baseline_value"] = baseline["value"]
            item["log_change"] = float(np.log(item["value"]) - np.log(baseline["value"]))
    return values


def write_outputs(output_dir, values, row, column):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "single_pixel_values.csv"
    raw_fields = [
        "led",
        "repeat",
        "group",
        "exposure_start_time",
        "days",
        "value",
        "dq",
        "baseline_value",
        "log_change",
    ]
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        for item in values:
            writer.writerow({field: item[field] for field in raw_fields})

    statistics_path = output_dir / "single_pixel_log_statistics.csv"
    statistics_fields = [
        "led",
        "repeat",
        "group",
        "file_set_name",
        "baseline_file_set_name",
        "exposure_start_time",
        "days",
        "pixel_count",
        "sum_log_change",
        "sum_squared_log_change",
        "mean_log_change",
        "pixel_rms_about_mean_log",
        "naive_standard_error_log",
        "min_log_change",
        "max_log_change",
    ]
    with statistics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=statistics_fields)
        writer.writeheader()
        for item in values:
            log_change = item["log_change"]
            writer.writerow(
                {
                    "led": item["led"],
                    "repeat": item["repeat"],
                    "group": item["group"],
                    "file_set_name": item["file_set_name"],
                    "baseline_file_set_name": next(
                        baseline["file_set_name"]
                        for baseline in values
                        if baseline["led"] == item["led"] and baseline["repeat"] == 0
                    ),
                    "exposure_start_time": item["exposure_start_time"],
                    "days": f"{item['days']:.12g}",
                    "pixel_count": 1,
                    "sum_log_change": f"{log_change:.17g}",
                    "sum_squared_log_change": f"{log_change * log_change:.17g}",
                    "mean_log_change": f"{log_change:.17g}",
                    "pixel_rms_about_mean_log": 0.0,
                    "naive_standard_error_log": 0.0,
                    "min_log_change": f"{log_change:.17g}",
                    "max_log_change": f"{log_change:.17g}",
                }
            )

    summary_path = output_dir / "single_pixel_selection.md"
    summary_path.write_text(
        f"# WFI01 single-pixel extraction\n\n"
        f"- Pixel row: `{row}`\n"
        f"- Pixel column: `{column}`\n"
        f"- Measurements: `{len(values)}` (six LEDs across 19 epochs)\n"
        f"- DQ status: all selected samples have `DQ=0`\n"
        f"- Baselines: each LED is normalized to its own first epoch\n"
    )
    return raw_path, statistics_path, summary_path


def main():
    args = parse_args()
    if args.row < 0 or args.column < 0:
        raise ValueError("Pixel coordinates must be non-negative")
    observations, _ = read_wfi01_index(args.index_path, args.data_dir)
    values = extract_pixel(observations, args.row, args.column)
    paths = write_outputs(args.output_dir, values, args.row, args.column)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
