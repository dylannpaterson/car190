from pathlib import Path
from unittest.mock import patch

import numpy as np

from reset_events import load_reset_config
from wfi_pipeline import run_wfi_global_rate_plots as global_plots
from wfi_pipeline.wfi_pipeline_defaults import DETECTORS, FULL_TMM


def test_global_rate_plot_dispatch_uses_all_detector_fits_and_reset_days(tmp_path):
    production_root = tmp_path / "production"
    reset_config = load_reset_config()
    days = np.asarray([12.907397928240743, 38.79784510416666])
    names = np.asarray([event["name"] for event in reset_config["events"]])
    for detector in DETECTORS:
        archive_path = (
            production_root
            / detector.lower()
            / "full_tmm"
            / "fits"
            / "tile_000"
            / "posterior_samples.npz"
        )
        archive_path.parent.mkdir(parents=True)
        np.savez_compressed(
            archive_path,
            reset_event_days=days,
            reset_event_names=names,
        )
    outputs = [str(tmp_path / "plots" / "focal_plane.png")]

    with patch.object(
        global_plots,
        "_run_focal_plane_and_ice_trend_plots",
        return_value=outputs,
    ) as plot:
        result = global_plots.build_global_rate_plots(
            tmp_path / "corrected_statistics",
            production_root,
            FULL_TMM,
            tmp_path / "plots",
        )

    assert result == outputs
    args = plot.call_args.args
    assert args[0] == tmp_path / "corrected_statistics"
    assert args[1] == tmp_path / "plots"
    assert set(args[2]) == set(DETECTORS)
    assert plot.call_args.kwargs["reset_day"] == 12.907397928240743
    np.testing.assert_allclose(plot.call_args.kwargs["reset_days"], days)
