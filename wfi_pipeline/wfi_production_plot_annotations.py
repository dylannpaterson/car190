"""Time-axis annotations shared by the production WFI plots."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")
SCE_TEMPERATURE_CHANGE_UTC = datetime(2026, 10, 17, 3, 30, tzinfo=UTC)
SCE_TEMPERATURE_CHANGE_LABEL = "SCE 153.5 to 155 K"
ENCLOSURE_TEMPERATURE_CHANGE_UTC = datetime(2026, 10, 17, 23, 0, tzinfo=UTC)
ENCLOSURE_TEMPERATURE_CHANGE_LABEL = "Enclosure temperature +4 K"
AOM_TEMPERATURE_BUMP_UTC = datetime(2026, 10, 30, 1, 43, tzinfo=UTC)
AOM_TEMPERATURE_BUMP_LABEL = "AOM temperature bump (01:43 UTC / 21:43 EDT Oct 29)"


@dataclass(frozen=True)
class Event:
    label: str
    day: float
    repeat: int
    kind: str


@dataclass(frozen=True)
class TimeAnnotations:
    start_utc: datetime
    physical_reset_day: float | None
    car216_intervals: tuple[tuple[float, float], ...]
    events: tuple[Event, ...]


def _parse_timestamp(value):
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _statistics_paths(statistics_path, tile_size=None):
    statistics_path = Path(statistics_path)
    if statistics_path.is_file():
        return [statistics_path]
    if not statistics_path.is_dir():
        return []
    pattern = "wfi*_tile_log_statistics_*.csv"
    if tile_size is not None:
        pattern = f"wfi*_tile_log_statistics_{int(tile_size)}*.csv"
    return sorted(statistics_path.glob(pattern))


def latest_observation_datetime(statistics_path, tile_size=None):
    """Return the latest exposure timestamp normalized to UTC."""
    latest = None
    for path in _statistics_paths(statistics_path, tile_size):
        for row in _read_rows(path):
            value = str(row.get("exposure_start_time", "")).strip()
            if not value:
                continue
            try:
                observation = _parse_timestamp(value)
            except ValueError:
                continue
            if latest is None or observation > latest:
                latest = observation
    return latest


def _format_timestamp(value):
    return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")


def latest_observation_label(statistics_path, tile_size=None):
    """Format the latest observation in both UTC and US Eastern time."""
    latest = latest_observation_datetime(statistics_path, tile_size)
    if latest is None:
        return "Latest observation: unknown"
    utc = latest.astimezone(UTC)
    eastern = latest.astimezone(EASTERN)
    return (
        f"Latest observation: {_format_timestamp(utc)} UTC / "
        f"{_format_timestamp(eastern)} {eastern.tzname()}"
    )


def _canonical_statistics_path(path):
    path = Path(path)
    detector, suffix = path.name.split("_tile_log_statistics_", 1)
    tile_size = suffix.split("_", 1)[0].split(".", 1)[0]
    canonical_name = (
        f"{detector.lower()}_tile_log_statistics_{tile_size}_count_corrected.csv"
    )
    for parent in path.parents:
        candidate = parent / "corrected_statistics" / canonical_name
        if candidate.is_file():
            return candidate
    return path


def _pipeline_manifest(path):
    for parent in (Path(path).parent, *Path(path).parents):
        candidate = parent / "pipeline_manifest.json"
        if candidate.is_file():
            with candidate.open() as handle:
                return json.load(handle)
    return {}


def _effective_event_repeats(path):
    detector = Path(path).name.split("_tile_log_statistics_", 1)[0].upper()
    manifest = _pipeline_manifest(path)
    detector_manifest = manifest.get("detectors", {}).get(detector, {})
    effective = detector_manifest.get("effective_correction_repeats", {})
    defaults = {
        "first_acm_step_repeat": 14,
        "pre_reset_acm_step_repeat": 35,
        "additional_acm_step_repeat": 39,
        "day22_acm_step_repeat": 52,
        "day26_acm_step_repeat": 61,
    }
    repeats = {
        name: int(effective.get(name, fallback))
        for name, fallback in defaults.items()
    }
    side_swap = detector_manifest.get("side_swap", {})
    return repeats, int(side_swap.get("event_repeat", 56))


def _repeat_chunks(rows, gap_hours=1.0):
    by_repeat = {}
    for row in rows:
        repeat = int(row["repeat"])
        by_repeat.setdefault(repeat, []).append(row)
    repeats = []
    for repeat, repeat_rows in by_repeat.items():
        timestamps = [
            _parse_timestamp(row["exposure_start_time"]) for row in repeat_rows
        ]
        repeats.append(
            (
                repeat,
                min(timestamps),
                max(timestamps),
                min(float(row["days"]) for row in repeat_rows),
                max(float(row["days"]) for row in repeat_rows),
            )
        )
    repeats.sort(key=lambda item: item[1])
    chunks = []
    current = []
    for item in repeats:
        if current and (item[1] - current[-1][2]).total_seconds() > gap_hours * 3600.0:
            chunks.append(current)
            current = []
        current.append(item)
    if current:
        chunks.append(current)
    return chunks


def load_time_annotations(statistics_path):
    """Load UTC, CAR216, and correction-event locations for a plot."""
    statistics_path = Path(statistics_path)
    rows = _read_rows(statistics_path)
    if not rows:
        raise ValueError(f"No rows found in {statistics_path}")
    start_utc = min(_parse_timestamp(row["exposure_start_time"]) for row in rows)
    car216_rows = [row for row in rows if str(row.get("program")) == "1063"]
    event_rows = _read_rows(_canonical_statistics_path(statistics_path))
    manifest = _pipeline_manifest(statistics_path)
    detector = statistics_path.name.split("_tile_log_statistics_", 1)[0].upper()
    detector_manifest = manifest.get("detectors", {}).get(detector, {})
    physical_reset = detector_manifest.get("physical_ice_reset", {})
    physical_reset_day = physical_reset.get("day")
    source_chunks = _repeat_chunks(
        [row for row in event_rows if str(row.get("program")) == "1063"]
    )
    car216_intervals = []
    for chunk in source_chunks:
        chunk_start = chunk[0][1]
        chunk_end = chunk[-1][2]
        selected = [
            row
            for row in car216_rows
            if chunk_start
            <= _parse_timestamp(row["exposure_start_time"])
            <= chunk_end
        ]
        if selected:
            car216_intervals.append(
                (
                    min(float(row["days"]) for row in selected),
                    max(float(row["days"]) for row in selected),
                )
            )
    acm_repeats, side_swap_repeat = _effective_event_repeats(statistics_path)
    event_definitions = (
        ("ACM step 1", acm_repeats["first_acm_step_repeat"], "acm"),
        ("ACM step 2", acm_repeats["pre_reset_acm_step_repeat"], "acm"),
        ("ACM step 3", acm_repeats["additional_acm_step_repeat"], "acm"),
        ("ACM step 4", acm_repeats["day22_acm_step_repeat"], "acm"),
        ("side swap", side_swap_repeat, "side_swap"),
        ("ACM step 5", acm_repeats["day26_acm_step_repeat"], "acm"),
    )
    events = []
    events.append(
        Event(
            label=SCE_TEMPERATURE_CHANGE_LABEL,
            day=(SCE_TEMPERATURE_CHANGE_UTC - start_utc).total_seconds() / 86400.0,
            repeat=-1,
            kind="temperature",
        )
    )
    events.append(
        Event(
            label=ENCLOSURE_TEMPERATURE_CHANGE_LABEL,
            day=(
                ENCLOSURE_TEMPERATURE_CHANGE_UTC - start_utc
            ).total_seconds()
            / 86400.0,
            repeat=-1,
            kind="temperature",
        )
    )
    events.append(
        Event(
            label=AOM_TEMPERATURE_BUMP_LABEL,
            day=(AOM_TEMPERATURE_BUMP_UTC - start_utc).total_seconds() / 86400.0,
            repeat=-1,
            kind="temperature",
        )
    )
    minimum_day = manifest.get("correction_settings", {}).get("minimum_day")
    if minimum_day is not None:
        events.append(
            Event(
                label="CAR190 temp gain ends (1039 day 1)",
                day=float(minimum_day),
                repeat=-1,
                kind="temperature",
            )
        )
    for label, repeat, kind in event_definitions:
        matches = [row for row in event_rows if int(row["repeat"]) == repeat]
        if not matches:
            matches = [row for row in event_rows if int(row["repeat"]) >= repeat]
        if not matches:
            continue
        event_row = min(matches, key=lambda row: _parse_timestamp(row["exposure_start_time"]))
        events.append(
            Event(
                label=label,
                day=float(event_row["days"]),
                repeat=repeat,
                kind=kind,
            )
        )
    events.sort(key=lambda event: event.day)
    return TimeAnnotations(
        start_utc=start_utc,
        physical_reset_day=(
            None if physical_reset_day is None else float(physical_reset_day)
        ),
        car216_intervals=tuple(car216_intervals),
        events=tuple(events),
    )


def _event_style(kind):
    if kind == "acm":
        return "#D55E00", "--"
    if kind == "side_swap":
        return "#7B3294", ":"
    return "#009E73", "-."


def event_legend_handles(annotations):
    handles = []
    if annotations.car216_intervals:
        handles.append(
            Patch(
                facecolor="#56B4E9",
                edgecolor="none",
                alpha=0.16,
                label="CAR216",
            )
        )
    for event in annotations.events:
        color, linestyle = _event_style(event.kind)
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
                label=event.label,
            )
        )
    return handles


def non_annotation_legend_handles(axis, annotations):
    handles, labels = axis.get_legend_handles_labels()
    if annotations is None:
        return handles, labels
    excluded = {"CAR216", *(event.label for event in annotations.events)}
    filtered = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label not in excluded
    ]
    if not filtered:
        return [], []
    return tuple(zip(*filtered))


def _days_to_utc_number(days, start_utc):
    values = np.asarray(days, dtype=float)
    flat = values.reshape(-1)
    mapped = np.asarray(
        [
            mdates.date2num(start_utc + timedelta(days=float(value)))
            for value in flat
        ],
        dtype=float,
    )
    return mapped.reshape(values.shape)


def _utc_tick_label(value, _position):
    timestamp = mdates.num2date(value, tz=UTC)
    return timestamp.strftime("%m-%d")


def _utc_number_to_days(numbers, start_utc):
    values = np.asarray(numbers, dtype=float)
    flat = values.reshape(-1)
    mapped = np.asarray(
        [
            (mdates.num2date(value, tz=UTC) - start_utc).total_seconds() / 86400.0
            for value in flat
        ],
        dtype=float,
    )
    return mapped.reshape(values.shape)


def annotate_time_axis(axis, annotations, date_tick_interval_days=1):
    """Add CAR216 shading, event lines, and a UTC secondary axis."""
    date_tick_interval_days = int(date_tick_interval_days)
    if date_tick_interval_days <= 0:
        raise ValueError("date_tick_interval_days must be positive")
    for index, (start_day, end_day) in enumerate(annotations.car216_intervals):
        axis.axvspan(
            start_day,
            end_day,
            color="#56B4E9",
            alpha=0.16,
            zorder=0,
            label="CAR216" if index == 0 else "_nolegend_",
        )
    for event in annotations.events:
        color, linestyle = _event_style(event.kind)
        axis.axvline(
            event.day,
            color=color,
            linestyle=linestyle,
            linewidth=0.9,
            alpha=0.9,
            label=event.label,
        )
    utc_axis = axis.secondary_xaxis(
        "top",
        functions=(
            lambda values: _days_to_utc_number(values, annotations.start_utc),
            lambda values: _utc_number_to_days(values, annotations.start_utc),
        ),
    )
    locator = mdates.DayLocator(interval=date_tick_interval_days, tz=UTC)
    utc_axis.xaxis.set_major_locator(locator)
    utc_axis.xaxis.set_major_formatter(FuncFormatter(_utc_tick_label))
    utc_axis.set_xlabel("UTC date (midnight)", labelpad=7)
    utc_axis.tick_params(axis="x", labelsize=7, pad=2)
    return utc_axis


def split_time_series_at_resets(days, *series, reset_days=None):
    """Insert NaN breaks at each requested reset boundary."""
    days = np.asarray(days, dtype=float)
    values = tuple(np.asarray(value, dtype=float) for value in series)
    if reset_days is None:
        return (days, *values)
    if np.isscalar(reset_days):
        reset_days = (reset_days,)
    split_indices = {
        int(np.count_nonzero(days < float(reset_day)))
        for reset_day in reset_days
    }
    split_indices = sorted(
        index for index in split_indices if 0 < index < days.size
    )
    for split_index in reversed(split_indices):
        days = np.insert(days, split_index, np.nan)
        values = tuple(
            np.insert(value, split_index, np.nan, axis=-1) for value in values
        )
    return (days, *values)


def split_time_series_at_reset(days, *series, reset_day=None):
    """Insert a NaN break between pre-reset and post-reset samples."""
    return split_time_series_at_resets(
        days,
        *series,
        reset_days=None if reset_day is None else (reset_day,),
    )
