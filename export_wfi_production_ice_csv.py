"""Export fitted WFI ice thickness and accumulation rates as CSV tables."""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path
import re

import numpy as np

from bundle_wfi_production_prior import (
    SCAS,
    find_fit_archives,
    load_statistics,
    statistics_fingerprint,
)
from wfi_production_layout import (
    DEFAULT_TILE_SIZE,
    corrected_statistics_path,
    positive_integer,
    production_output_root,
)
from wfi_wavelength_response_bundle import tile_geometry


COMMON_FIELDS = (
    "detector",
    "sca_number",
    "tile_id",
    "tile_row",
    "tile_col",
    "mpa_x_mm",
    "mpa_y_mm",
    "tile_local_x_mm",
    "tile_local_y_mm",
    "row_start",
    "row_stop_exclusive",
    "col_start",
    "col_stop_exclusive",
)

CURRENT_FIELDS = COMMON_FIELDS + (
    "latest_node_index",
    "repeat",
    "node_mjd_utc",
    "days_since_first_exposure",
    "ice_thickness_nm",
    "accumulation_rate_nm_day",
    "initial_ice_thickness_nm",
    "initial_accumulation_rate_nm_day",
    "final_accumulation_rate_nm_day",
    "mean_accumulation_rate_nm_day",
    "ice_reset_day",
    "third_decon_day",
    "post_reset",
    "post_third_decon",
    "fit_mode",
    "constant_rate_map_warmup",
    "fit_archive",
)

NODE_FIELDS = COMMON_FIELDS + (
    "node_index",
    "repeat",
    "node_mjd_utc",
    "days_since_first_exposure",
    "ice_thickness_nm",
    "accumulation_rate_nm_day",
    "ice_reset_day",
    "third_decon_day",
    "post_reset",
    "post_third_decon",
    "fit_mode",
    "constant_rate_map_warmup",
    "fit_archive",
)

DATE_LABEL_PATTERN = re.compile(r"\d{8}")


def infer_date_label(statistics_dir):
    """Return the nearest YYYYMMDD directory component in a run path."""
    for component in reversed(Path(statistics_dir).resolve().parts):
        if DATE_LABEL_PATTERN.fullmatch(component):
            return component
    raise ValueError(
        "Could not infer a YYYYMMDD run date from --statistics-dir; "
        "pass --date-label explicitly"
    )


def _numeric_scalar(samples, name, default=np.nan):
    if name not in samples:
        return default
    values = np.asarray(samples[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected scalar archive field {name}")
    return float(values[0])


def _text_scalar(samples, name, default=""):
    if name not in samples:
        return default
    values = np.asarray(samples[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected scalar archive field {name}")
    value = values[0].item() if isinstance(values[0], np.generic) else values[0]
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _series(samples, names, expected_size):
    for name in names:
        if name not in samples:
            continue
        values = np.asarray(samples[name], dtype=float).reshape(-1)
        if values.size != expected_size:
            raise ValueError(
                f"Unexpected {name} length: expected {expected_size}, got {values.size}"
            )
        return values
    raise KeyError(f"Missing archive field; tried {names}")


def _load_fit_archive(
    path, expected_days, expected_node_days, expected_fingerprint=None
):
    with np.load(path / "posterior_samples.npz", allow_pickle=False) as samples:
        if expected_fingerprint is not None:
            archive_fingerprint = _text_scalar(samples, "statistics_fingerprint")
            if archive_fingerprint != expected_fingerprint:
                raise ValueError(f"Statistics fingerprint mismatch for {path}")
        days = np.asarray(samples["observation_days"], dtype=float).reshape(-1)
        if days.size != expected_days.size or not np.allclose(
            days, expected_days, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError(f"Observation days differ in {path}")
        thickness = _series(samples, ("tile_ice_thickness", "ice_thickness"), days.size)
        rates = _series(
            samples,
            ("tile_instantaneous_rate_nm_day", "instantaneous_rate_nm_day"),
            days.size,
        )
        node_thickness = _series(
            samples,
            ("np_node_thickness", "tile_node_thickness", "node_ice_thickness"),
            expected_node_days.size,
        )
        node_rates = _series(
            samples,
            (
                "node_instantaneous_rate_nm_day",
                "tile_node_instantaneous_rate_nm_day",
            ),
            expected_node_days.size,
        )
        reset_day = _numeric_scalar(samples, "ice_reset_day")
        third_day = _numeric_scalar(samples, "third_decon_day")
        initial_log = _numeric_scalar(samples, "initial_thickness_log_nm")
        fit_warmup = int(_numeric_scalar(samples, "constant_rate_map_warmup", 0.0))
        return {
            "days": days,
            "thickness": thickness,
            "rates": rates,
            "node_thickness": node_thickness,
            "node_rates": node_rates,
            "ice_reset_day": reset_day,
            "third_decon_day": third_day,
            "initial_thickness_nm": (
                float(np.exp(initial_log)) if np.isfinite(initial_log) else np.nan
            ),
            "fit_mode": _text_scalar(samples, "fit_mode", "unknown"),
            "constant_rate_map_warmup": fit_warmup,
        }


def _geometry_row(detector, tile_id, metadata):
    geometry = tile_geometry(
        detector,
        tile_id,
        metadata["tile_row"],
        metadata["tile_col"],
        metadata["row_start"],
        metadata["row_stop_exclusive"],
        metadata["col_start"],
        metadata["col_stop_exclusive"],
    )
    return {
        "detector": detector,
        "sca_number": geometry["sca_number"],
        "tile_id": tile_id,
        "tile_row": metadata["tile_row"],
        "tile_col": metadata["tile_col"],
        "mpa_x_mm": round(float(geometry["tile_center_x_mm"]), 6),
        "mpa_y_mm": round(float(geometry["tile_center_y_mm"]), 6),
        "tile_local_x_mm": round(float(geometry["tile_local_x_mm"]), 6),
        "tile_local_y_mm": round(float(geometry["tile_local_y_mm"]), 6),
        "row_start": metadata["row_start"],
        "row_stop_exclusive": metadata["row_stop_exclusive"],
        "col_start": metadata["col_start"],
        "col_stop_exclusive": metadata["col_stop_exclusive"],
    }


def _archive_path(path, fit_root):
    return str(Path(path).resolve().relative_to(Path(fit_root).resolve()))


def _post_third_decon(day, third_day):
    return bool(np.isfinite(third_day) and day >= third_day)


def _post_reset(day, reset_day):
    return bool(np.isfinite(reset_day) and day >= reset_day)


def _trajectory_rows(common, fit, node_days, node_repeats, first_mjd, archive_name):
    for node_index, day_value in enumerate(node_days):
        day = float(day_value)
        yield {
            **common,
            "node_index": node_index,
            "repeat": int(node_repeats[node_index]),
            "node_mjd_utc": float(first_mjd + day),
            "days_since_first_exposure": day,
            "ice_thickness_nm": float(fit["node_thickness"][node_index]),
            "accumulation_rate_nm_day": float(fit["node_rates"][node_index]),
            "ice_reset_day": fit["ice_reset_day"],
            "third_decon_day": fit["third_decon_day"],
            "post_reset": _post_reset(day, fit["ice_reset_day"]),
            "post_third_decon": _post_third_decon(day, fit["third_decon_day"]),
            "fit_mode": fit["fit_mode"],
            "constant_rate_map_warmup": fit["constant_rate_map_warmup"],
            "fit_archive": archive_name,
        }


def _expected_fingerprint(source_rows):
    if source_rows and all(
        field in source_rows[0]
        for field in ("n_pair_valid", "sum_log_change", "sum_squared_log_change")
    ):
        return statistics_fingerprint(source_rows)
    return None


def _iter_tile_rows(
    statistics_dir,
    fit_root,
    detectors=SCAS,
    tile_size=DEFAULT_TILE_SIZE,
):
    """Yield one current row and a lazy model-node iterator per fitted tile."""
    statistics_dir = Path(statistics_dir).resolve()
    fit_root = Path(fit_root).resolve()

    for detector in detectors:
        statistics_path = corrected_statistics_path(statistics_dir, detector, tile_size)
        statistics = load_statistics(statistics_path, detector)
        expected_days = np.asarray(
            [
                observation["time_days_since_first_exposure"]
                for observation in statistics["observations"]
            ],
            dtype=float,
        )
        node_days = np.asarray(statistics["node_days"], dtype=float)
        node_repeats = np.asarray(
            sorted({observation["repeat"] for observation in statistics["observations"]}),
            dtype=int,
        )
        if node_days.size != node_repeats.size:
            raise ValueError(f"Node/repeat count differs for {detector}")
        archives, _layout = find_fit_archives(
            fit_root,
            detector,
            tile_size=tile_size,
            expected_tile_ids=statistics["tile_ids"],
        )
        for tile_id in statistics["tile_ids"]:
            archive_path = archives[tile_id]
            fit = _load_fit_archive(
                archive_path,
                expected_days,
                node_days,
                expected_fingerprint=_expected_fingerprint(
                    statistics["rows_by_tile"][tile_id]
                ),
            )
            metadata = statistics["tile_metadata"][tile_id]
            common = _geometry_row(detector, tile_id, metadata)
            archive_name = _archive_path(
                archive_path / "posterior_samples.npz", fit_root
            )
            initial_rate = float(fit["node_rates"][0])
            final_rate = float(fit["node_rates"][-1])
            mean_rate = float(np.mean(fit["node_rates"]))
            latest_node_index = node_days.size - 1
            latest_day = float(node_days[latest_node_index])
            yield (
                {
                    **common,
                    "latest_node_index": latest_node_index,
                    "repeat": int(node_repeats[latest_node_index]),
                    "node_mjd_utc": float(statistics["first_mjd"] + latest_day),
                    "days_since_first_exposure": latest_day,
                    "ice_thickness_nm": float(
                        fit["node_thickness"][latest_node_index]
                    ),
                    "accumulation_rate_nm_day": float(
                        fit["node_rates"][latest_node_index]
                    ),
                    "initial_ice_thickness_nm": fit["initial_thickness_nm"],
                    "initial_accumulation_rate_nm_day": initial_rate,
                    "final_accumulation_rate_nm_day": final_rate,
                    "mean_accumulation_rate_nm_day": mean_rate,
                    "ice_reset_day": fit["ice_reset_day"],
                    "third_decon_day": fit["third_decon_day"],
                    "post_reset": _post_reset(latest_day, fit["ice_reset_day"]),
                    "post_third_decon": _post_third_decon(
                        latest_day, fit["third_decon_day"]
                    ),
                    "fit_mode": fit["fit_mode"],
                    "constant_rate_map_warmup": fit["constant_rate_map_warmup"],
                    "fit_archive": archive_name,
                },
                _trajectory_rows(
                    common,
                    fit,
                    node_days,
                    node_repeats,
                    statistics["first_mjd"],
                    archive_name,
                ),
            )


def collect_rows(
    statistics_dir,
    fit_root,
    detectors=SCAS,
    tile_size=DEFAULT_TILE_SIZE,
    include_trajectory=True,
):
    """Collect current and shared model-node rows from completed fit archives."""
    current_rows = []
    trajectory_rows = []
    for current_row, tile_trajectory in _iter_tile_rows(
        statistics_dir, fit_root, detectors=detectors, tile_size=tile_size
    ):
        current_rows.append(current_row)
        if include_trajectory:
            trajectory_rows.extend(tile_trajectory)
    return current_rows, trajectory_rows


def _csv_value(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def write_rows(path, rows, fields):
    if not rows:
        raise ValueError("No rows to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def write_export(
    statistics_dir,
    fit_root,
    trajectory_path,
    current_path=None,
    detectors=SCAS,
    tile_size=DEFAULT_TILE_SIZE,
):
    """Stream shared model nodes and optionally write a latest-node snapshot."""
    trajectory_path = Path(trajectory_path)
    if current_path is not None:
        current_path = Path(current_path)
        if trajectory_path.resolve() == current_path.resolve():
            raise ValueError("Trajectory and current CSV paths must differ")
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    if current_path is not None:
        current_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_count = 0
    current_count = 0
    with ExitStack() as stack:
        trajectory_handle = stack.enter_context(
            trajectory_path.open("w", newline="")
        )
        trajectory_writer = csv.DictWriter(
            trajectory_handle, fieldnames=NODE_FIELDS
        )
        trajectory_writer.writeheader()
        current_writer = None
        if current_path is not None:
            current_handle = stack.enter_context(current_path.open("w", newline=""))
            current_writer = csv.DictWriter(current_handle, fieldnames=CURRENT_FIELDS)
            current_writer.writeheader()

        for current_row, tile_trajectory in _iter_tile_rows(
            statistics_dir, fit_root, detectors=detectors, tile_size=tile_size
        ):
            if current_writer is not None:
                current_writer.writerow(
                    {
                        field: _csv_value(current_row.get(field, ""))
                        for field in CURRENT_FIELDS
                    }
                )
                current_count += 1
            for trajectory_row in tile_trajectory:
                trajectory_writer.writerow(
                    {
                        field: _csv_value(trajectory_row.get(field, ""))
                        for field in NODE_FIELDS
                    }
                )
                trajectory_count += 1
    return current_count, trajectory_count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=None,
        help="Root containing completed production fit archives; defaults to statistics-dir.",
    )
    parser.add_argument("--tile-size", type=positive_integer, default=DEFAULT_TILE_SIZE)
    parser.add_argument(
        "--date-label",
        default=None,
        help="YYYYMMDD label for default output filenames; inferred from statistics-dir.",
    )
    parser.add_argument(
        "--detector",
        dest="detectors",
        action="append",
        choices=SCAS,
        default=None,
        help="Restrict export to selected detector(s); may be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="All-epoch trajectory CSV; defaults to the production output directory.",
    )
    parser.add_argument(
        "--current-output",
        type=Path,
        default=None,
        help="Optional latest-epoch snapshot CSV.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    detectors = tuple(args.detectors or SCAS)
    fit_root = args.fit_root or args.statistics_dir
    date_label = args.date_label or infer_date_label(args.statistics_dir)
    if not DATE_LABEL_PATTERN.fullmatch(date_label):
        raise ValueError(f"Invalid date label {date_label!r}; expected YYYYMMDD")
    if args.output is not None and args.trajectory_output is not None:
        raise ValueError("Use only one of --output and --trajectory-output")
    trajectory_output = args.output or args.trajectory_output or (
        production_output_root(args.statistics_dir, args.tile_size)
        / "relative_response"
        / f"wfi_ice_accumulation_{date_label}.csv"
    )
    current_count, trajectory_count = write_export(
        args.statistics_dir,
        fit_root,
        trajectory_path=trajectory_output,
        current_path=args.current_output,
        detectors=detectors,
        tile_size=args.tile_size,
    )
    print(f"Wrote {trajectory_count} model-node rows to {trajectory_output}")
    if args.current_output is not None:
        print(f"Wrote {current_count} current tile rows to {args.current_output}")


if __name__ == "__main__":
    main()
