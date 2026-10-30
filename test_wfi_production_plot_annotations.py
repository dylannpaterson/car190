import csv

from wfi_production_plot_annotations import (
    AOM_TEMPERATURE_BUMP_LABEL,
    ENCLOSURE_TEMPERATURE_CHANGE_LABEL,
    SCE_TEMPERATURE_CHANGE_LABEL,
    latest_observation_label,
    load_time_annotations,
)


def _write_statistics(path, timestamp):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["exposure_start_time", "program", "repeat", "days"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "exposure_start_time": timestamp,
                "program": "1039",
                "repeat": "0",
                "days": "0.0",
            }
        )


def test_known_sce_temperature_change_is_loaded_as_event(tmp_path):
    corrected = tmp_path / "corrected_statistics"
    corrected.mkdir()
    canonical = corrected / "wfi01_tile_log_statistics_256_count_corrected.csv"
    _write_statistics(canonical, "2026-10-17T01:30:00+00:00")

    statistics = tmp_path / "thinned" / "wfi01_tile_log_statistics_256.csv"
    statistics.parent.mkdir()
    _write_statistics(statistics, "2026-10-17T01:30:00+00:00")

    annotations = load_time_annotations(statistics)
    event = next(
        event for event in annotations.events if event.label == SCE_TEMPERATURE_CHANGE_LABEL
    )
    assert event.kind == "temperature"
    assert event.repeat == -1
    assert event.day == 2.0 / 24.0


def test_enclosure_temperature_change_is_loaded_as_event(tmp_path):
    corrected = tmp_path / "corrected_statistics"
    corrected.mkdir()
    canonical = corrected / "wfi01_tile_log_statistics_256_count_corrected.csv"
    _write_statistics(canonical, "2026-10-17T00:00:00+00:00")

    statistics = tmp_path / "thinned" / "wfi01_tile_log_statistics_256.csv"
    statistics.parent.mkdir()
    _write_statistics(statistics, "2026-10-17T00:00:00+00:00")

    annotations = load_time_annotations(statistics)
    event = next(
        event
        for event in annotations.events
        if event.label == ENCLOSURE_TEMPERATURE_CHANGE_LABEL
    )
    assert event.kind == "temperature"
    assert event.repeat == -1
    assert event.day == 23.0 / 24.0


def test_aom_temperature_bump_is_loaded_as_event(tmp_path):
    corrected = tmp_path / "corrected_statistics"
    corrected.mkdir()
    canonical = corrected / "wfi01_tile_log_statistics_256_count_corrected.csv"
    _write_statistics(canonical, "2026-10-29T00:00:00+00:00")

    statistics = tmp_path / "thinned" / "wfi01_tile_log_statistics_256.csv"
    statistics.parent.mkdir()
    _write_statistics(statistics, "2026-10-29T00:00:00+00:00")

    annotations = load_time_annotations(statistics)
    event = next(
        event for event in annotations.events if event.label == AOM_TEMPERATURE_BUMP_LABEL
    )
    assert event.kind == "temperature"
    assert event.repeat == -1
    assert event.day == (24 * 60 + 103) / (24 * 60)


def test_latest_observation_label_includes_utc_and_edt(tmp_path):
    path = tmp_path / "wfi01_tile_log_statistics_256_count_corrected.csv"
    _write_statistics(path, "2026-10-18T08:03:56.517000+00:00")

    assert latest_observation_label(tmp_path, tile_size=256) == (
        "Latest observation: 2026-10-18 08:03:56.517 UTC / "
        "2026-10-18 04:03:56.517 EDT"
    )
