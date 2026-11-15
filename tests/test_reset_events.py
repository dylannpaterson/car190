import numpy as np
import pytest

from reset_events import load_reset_config, resolve_reset_events
import fit_wfi_tmm_production as production


def test_repository_reset_config_preserves_validated_boundary_and_priors():
    config = load_reset_config()

    assert [event["name"] for event in config["events"]] == [
        "second_decon",
        "third_decon",
    ]
    assert config["events"][0]["day"] == pytest.approx(12.907397928240743)
    assert config["events"][1]["day"] == pytest.approx(38.79784510416666)
    assert config["events"][1]["relative_response_warmup"] == {
        "start_nm": 40.0,
        "log_sigma": 0.1,
    }
    assert config["events"][1]["h5_prior"] == {
        "center": "base_h5",
        "log_sigma": 0.1,
    }


def test_reset_site_names_support_three_configured_decons():
    data = {
        "reset_days": np.asarray([12.0, 38.0, 55.0]),
        "reset_event_names": (
            "second_decon",
            "third_decon",
            "fourth_decon",
        ),
    }

    thickness_names, offset_names = production.reset_site_names(data)
    assert thickness_names == (
        "initial_thickness_log_nm",
        "reset_initial_thickness_log_nm",
        "third_decon_initial_thickness_log_nm",
        "reset_event_3_initial_thickness_log_nm",
    )
    assert offset_names == (
        "post_reset_tile_log_offset",
        "post_third_decon_tile_log_offset",
        "post_reset_event_3_tile_log_offset",
    )
    values = production.initial_values_for_data(
        {**data, "n_scas": 1, "n_tiles": 1, "node_days": np.arange(60.0)},
        fit_incidence_angle=False,
    )
    assert all(name in values for name in thickness_names + offset_names)


def test_configured_reset_events_keep_the_explicit_decon_days():
    config = load_reset_config()
    events = resolve_reset_events(config, 12.0)

    np.testing.assert_allclose(
        [event["day"] for event in events],
        [12.907397928240743, 38.79784510416666],
    )
    events_for_different_statistics = resolve_reset_events(config, 40.0)
    np.testing.assert_allclose(
        [event["day"] for event in events_for_different_statistics],
        [12.907397928240743, 38.79784510416666],
    )


def test_tile_statistics_loads_an_arbitrary_yaml_reset_list(tmp_path):
    config_path = tmp_path / "resets.yaml"
    config_path.write_text(
        "version: 1\n"
        "events:\n"
        "  - name: second_decon\n"
        "    day_source: program1063\n"
        "    thickness_prior: {median_nm: 10, log_sigma: 1.5}\n"
        "  - name: fourth_decon\n"
        "    day: 20\n"
        "    thickness_prior: {median_nm: 15, log_sigma: 0.7}\n"
        "  - name: fifth_decon\n"
        "    day: 30\n"
        "    thickness_prior: {median_nm: 25, log_sigma: 0.4}\n"
    )
    config = load_reset_config(config_path)
    stats_path = tmp_path / "wfi01_tile_log_statistics_256_count_corrected.csv"
    rows = [
        "tile_id,group,n_pair_valid,sum_log_change,sum_squared_log_change,led,repeat,program,days,exposure_start_time"
    ]
    group = 0
    for repeat, day in enumerate((10.0, 12.0, 16.0, 23.0, 35.0)):
        for led in range(1, 7):
            program = 1063 if repeat == 1 else 1039
            rows.append(
                f"0,{group},100,0,0,{led},{repeat},{program},{day},2026-10-01T00:00:00"
            )
            group += 1
    stats_path.write_text("\n".join(rows) + "\n")

    data = production.load_tile_statistics(
        stats_path, reset_events=config["events"]
    )

    np.testing.assert_allclose(data["reset_days"], [12.0, 20.0, 30.0])
    assert data["reset_event_names"] == (
        "second_decon",
        "fourth_decon",
        "fifth_decon",
    )
    assert production.reset_site_names(data)[0][-1] == (
        "reset_event_3_initial_thickness_log_nm"
    )
