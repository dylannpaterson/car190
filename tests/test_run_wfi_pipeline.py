import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from wfi_pipeline import run_wfi_pipeline as pipeline
import run_wfi_full_tmm_pipeline
import run_wfi_relative_response_pipeline
import fit_wfi_tmm_production as tmm_production
import run_wfi_tmm_emcee_warmup as emcee_warmup
from wfi_pipeline_defaults import (
    DETECTORS,
    FULL_TMM_EMCEE_SEED,
    FULL_TMM_EMCEE_STEPS,
    FULL_TMM_EMCEE_WALKERS,
    FULL_TMM_MAP_FTOL,
    FULL_TMM_MAP_MAX_EVALUATIONS,
    FULL_TMM_MAP_XTOL,
    FULL_TMM,
    RELATIVE_RESPONSE,
    detector_fit_output_root,
    detector_six_led_cache_root,
    output_paths,
    pinned_relative_response_cache,
    profile_for,
)
from reset_events import load_reset_config


class WfiPipelineTests(unittest.TestCase):
    def make_statistics(self, data_root):
        statistics = data_root / "corrected_statistics_4h_thinned"
        statistics.mkdir(parents=True)
        for detector in DETECTORS:
            (statistics / f"{detector.lower()}_tile_log_statistics_256_count_corrected.csv").write_text(
                "tile_id,group\n"
            )
        (statistics / "program1063_cadence_thinning_manifest.json").write_text(
            json.dumps({"policy": {"spacing_hours": 4.0, "tile_size": 256}})
        )
        return statistics

    def test_profiles_preserve_distinct_cache_policies(self):
        relative = profile_for(RELATIVE_RESPONSE)
        full_tmm = profile_for(FULL_TMM)
        self.assertTrue(relative.use_pinned_six_led_cache)
        self.assertFalse(relative.build_six_led_cache_from_current_tmm)
        self.assertFalse(full_tmm.use_pinned_six_led_cache)
        self.assertTrue(full_tmm.build_six_led_cache_from_current_tmm)

    def test_pipeline_records_and_passes_the_shared_reset_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = pipeline.PipelineRun(
                root / "corrected_statistics_4h_thinned",
                root / "production_run",
                FULL_TMM,
            )
            reset_config = load_reset_config()
            settings = run._settings_record()
            self.assertEqual(
                settings["reset_configuration"]["sha256"], reset_config["sha256"]
            )
            self.assertEqual(
                [event["name"] for event in settings["reset_configuration"]["events"]],
                [event["name"] for event in reset_config["events"]],
            )
            with (
                patch.object(run, "_run_command") as run_command,
                patch.object(run, "_tile_archives", return_value=True),
            ):
                run._fit_detector(root / "corrected_statistics_4h_thinned", "WFI01")
            command = run_command.call_args.args[1]
            config_index = command.index("--reset-config")
            self.assertEqual(command[config_index + 1], str(reset_config["path"]))

    def test_relative_pipeline_applies_warmup_settings_for_many_reset_events(self):
        events = (
            {
                "name": "second_decon",
                "day": 12.0,
                "thickness_prior": {"median_nm": 10.0, "log_sigma": 1.5},
            },
            {
                "name": "third_decon",
                "day": 38.0,
                "thickness_prior": {"median_nm": 10.0, "log_sigma": 1.5},
                "relative_response_warmup": {"start_nm": 40.0, "log_sigma": 0.1},
            },
            {
                "name": "fourth_decon",
                "day": 55.0,
                "thickness_prior": {"median_nm": 20.0, "log_sigma": 0.8},
                "relative_response_warmup": {"start_nm": 25.0, "log_sigma": 0.25},
            },
        )
        config = {
            "path": Path("/tmp/reset_events_test.yaml"),
            "sha256": "test-reset-sha256",
            "events": events,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(pipeline, "load_reset_config", return_value=config):
                run = pipeline.PipelineRun(
                    root / "corrected_statistics_4h_thinned",
                    root / "production_run",
                    RELATIVE_RESPONSE,
                )
                with (
                    patch.object(run, "_cache_complete", return_value=True),
                    patch.object(run, "_tile_archives", return_value=True),
                    patch.object(run, "_run_command") as run_command,
                ):
                    run._fit_detector(
                        root / "corrected_statistics_4h_thinned", "WFI01"
                    )

        commands = [call.args[1] for call in run_command.call_args_list]
        assert len(commands) == 2  # constant-rate start, then final cached fit
        warmup = commands[0]
        fit = commands[1]
        assert warmup[warmup.index("--reset-config") + 1] == str(config["path"])
        assert fit[fit.index("--reset-config") + 1] == str(config["path"])
        assert fit[fit.index("--thickness-prior-stages") + 1] == "configured"
        assert "--thickness-prior-log-sigma" not in fit
        assert run.reset_warmup_starts == {
            "third_decon": 40.0,
            "fourth_decon": 25.0,
        }

    def test_main_pipeline_dispatches_joint_modulated_trend_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = pipeline.PipelineRun(
                root / "corrected_statistics_4h_thinned",
                root / "production_run",
                FULL_TMM,
            )
            with patch.object(run, "_run_command") as run_command:
                bundle = root / "wavelength_response_bundle" / "modulated.npz"
                summary = run._run_joint_modulated_trend_plots(
                    root / "corrected_statistics_4h_thinned", bundle
                )
            command = run_command.call_args.args[1]
            assert str(Path("plot_joint_modulated_ice_trends.py")) in " ".join(command)
            assert command[command.index("--bundle-path") + 1] == str(bundle)
            assert command[command.index("--statistics-dir") + 1] == str(
                root / "corrected_statistics_4h_thinned"
            )
            outputs = run_command.call_args.kwargs["outputs"]
            assert outputs == [summary]
            assert summary.name == "joint_modulated_ice_trend_plots.json"

    def test_relative_route_fit_plots_do_not_claim_joint_history_trends(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = pipeline.PipelineRun(
                root / "corrected_statistics_4h_thinned",
                root / "production_run",
                RELATIVE_RESPONSE,
            )
            fit_root = root / "production_run" / "production" / "wfi01" / "relative_response"
            with patch.object(run, "_run_command") as run_command:
                run._run_tmm_plots(root / "corrected_statistics_4h_thinned", "WFI01", fit_root)
            command = run_command.call_args.args[1]
            assert "--fit-diagnostics-only" in command
            outputs = run_command.call_args.kwargs["outputs"]
            assert not any("trajectory" in path.name or "rate_maps" in path.name for path in outputs)

    def test_central_full_tmm_defaults_match_fit_runners(self):
        self.assertEqual(FULL_TMM_EMCEE_WALKERS, emcee_warmup.EMCEE_WALKERS)
        self.assertEqual(FULL_TMM_EMCEE_STEPS, emcee_warmup.EMCEE_STEPS)
        self.assertEqual(FULL_TMM_EMCEE_SEED, emcee_warmup.EMCEE_SEED)

    def test_emcee_site_order_has_unique_sites_for_arbitrary_reset_count(self):
        events = [
            {"name": name, "day": float(index + 1)}
            for index, name in enumerate(
                ("second_decon", "third_decon", "fourth_decon", "fifth_decon")
            )
        ]

        site_order = emcee_warmup.reset_aware_site_order(
            emcee_warmup.SITE_ORDER, events
        )

        assert len(site_order) == len(set(site_order))
        assert "reset_event_3_initial_thickness_log_nm" in site_order
        assert "post_reset_event_3_tile_log_offset" in site_order
        self.assertEqual(FULL_TMM_MAP_MAX_EVALUATIONS, tmm_production.PRODUCTION_MAP_STEPS)
        self.assertEqual(FULL_TMM_MAP_FTOL, tmm_production.PRODUCTION_MAP_FTOL)
        self.assertEqual(FULL_TMM_MAP_XTOL, tmm_production.PRODUCTION_MAP_XTOL)

    def test_cache_and_fit_paths_are_route_specific(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "20261113" / "production_run"
            self.assertEqual(
                detector_six_led_cache_root(output, RELATIVE_RESPONSE, "WFI03"),
                pinned_relative_response_cache("WFI03"),
            )
            self.assertEqual(
                detector_six_led_cache_root(output, FULL_TMM, "WFI03"),
                output_paths(output, FULL_TMM)["six_led_cache_root"] / "wfi03",
            )
            self.assertEqual(
                detector_fit_output_root(output, RELATIVE_RESPONSE, "WFI03"),
                output / "production" / "wfi03" / "relative_response",
            )
            self.assertEqual(
                detector_fit_output_root(output, FULL_TMM, "WFI03"),
                output / "production" / "wfi03" / "full_tmm",
            )

    def test_cached_route_spectral_grid_uses_original_tmm_optical_fits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_fit_root = root / "pipeline" / "production" / "wfi01" / "relative_response"
            cache_root = root / "20261109" / "pinned_cache" / "wfi01_400nm"
            cache_file = cache_root / "tile_000" / "relative_response.npz"
            cache_file.parent.mkdir(parents=True)
            cache_file.touch()
            source_fit = (
                root / "20261109" / "production" / "wfi01" / "fits"
                / "tile_000" / "posterior_samples.npz"
            )
            fake_run = SimpleNamespace(profile=profile_for(RELATIVE_RESPONSE))
            with patch(
                "run_wfi_relative_response_production.cache_source_paths",
                return_value=(source_fit, root / "source_statistics.csv"),
            ):
                spectral_fit_root = pipeline.PipelineRun._spectral_cache_fit_root(
                    fake_run, output_fit_root, cache_root
                )
            self.assertEqual(spectral_fit_root, source_fit.parent.parent)

            full_tmm_run = SimpleNamespace(profile=profile_for(FULL_TMM))
            self.assertEqual(
                pipeline.PipelineRun._spectral_cache_fit_root(
                    full_tmm_run, output_fit_root, cache_root
                ),
                output_fit_root / "fits",
            )

    def test_output_layout_puts_logs_at_run_root(self):
        paths = output_paths(Path("/data/20261113/run"), RELATIVE_RESPONSE)
        self.assertEqual(paths["logs"], Path("/data/20261113/run/logs"))
        self.assertEqual(paths["manifest"], Path("/data/20261113/run/pipeline_manifest.json"))
        self.assertEqual(
            paths["h5_root"],
            Path("/data/20261113/run/production/wavelength_response_bundle/final"),
        )
        self.assertEqual(
            paths["h5_intermediate_root"],
            Path("/data/20261113/run/production/wavelength_response_bundle/intermediates/h5"),
        )

    def test_thinning_manifest_must_match_pipeline_cadence_and_tile_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            statistics = self.make_statistics(Path(temporary) / "20261113")
            manifest = statistics / "program1063_cadence_thinning_manifest.json"
            manifest.write_text(
                json.dumps({"policy": {"spacing_hours": 1.0, "tile_size": 256}})
            )
            with self.assertRaisesRegex(ValueError, "expected 4 hours"):
                pipeline.resolve_input_statistics(statistics)

    def test_frontends_dispatch_only_their_named_route(self):
        args = ["--input-path", "/data/20261113", "--output-path", "/data/20261113/run"]
        with patch("wfi_pipeline.run_wfi_pipeline.PipelineRun") as run:
            run_wfi_relative_response_pipeline.main(args)
            self.assertEqual(run.call_args.args[2], RELATIVE_RESPONSE)
            run.return_value.run.assert_called_once_with()
        with patch("wfi_pipeline.run_wfi_pipeline.PipelineRun") as run:
            run_wfi_full_tmm_pipeline.main(args)
            self.assertEqual(run.call_args.args[2], FULL_TMM)
            run.return_value.run.assert_called_once_with()

    def test_initialize_allows_output_outside_input_root_but_not_inside_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "20261113"
            statistics = self.make_statistics(data_root)
            output = data_root / "production" / "pipeline_runs" / "relative"
            run = pipeline.PipelineRun(statistics, output, RELATIVE_RESPONSE)
            self.assertTrue(run.initialize())
            self.assertEqual(len(run.manifest["input_fingerprints"]), 19)
            self.assertTrue((output / "pipeline_manifest.json").is_file())
            self.assertEqual(run.paths["logs"], output / "logs")

            outside = Path(temporary) / "modeling" / "20261113"
            separate_run = pipeline.PipelineRun(statistics, outside, RELATIVE_RESPONSE)
            self.assertTrue(separate_run.initialize())
            self.assertEqual(len(separate_run.manifest["input_fingerprints"]), 19)

            inside_statistics = statistics / "production_run"
            with self.assertRaisesRegex(ValueError, "inside the corrected-statistics directory"):
                pipeline.PipelineRun(
                    statistics, inside_statistics, RELATIVE_RESPONSE
                ).initialize()

    def test_resume_rejects_changed_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "20261113"
            statistics = self.make_statistics(data_root)
            output = data_root / "production" / "pipeline_runs" / "relative"
            pipeline.PipelineRun(statistics, output, RELATIVE_RESPONSE).initialize()
            source_file = statistics / "wfi01_tile_log_statistics_256_count_corrected.csv"
            source_file.write_text("tile_id,group\n0,0\n")
            with self.assertRaisesRegex(ValueError, "changed statistics"):
                pipeline.PipelineRun(statistics, output, RELATIVE_RESPONSE).initialize()

    def test_stage_logs_are_written_under_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "20261113"
            statistics = self.make_statistics(data_root)
            output = data_root / "production" / "pipeline_runs" / "relative"
            run = pipeline.PipelineRun(statistics, output, RELATIVE_RESPONSE)
            run.initialize()
            with patch.object(pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                run._run_command("fixture_stage", ["python", "fixture.py"])
            log = output / "logs" / "fixture_stage" / "all.log"
            self.assertTrue(log.is_file())
            self.assertIn("fixture.py", log.read_text())


if __name__ == "__main__":
    unittest.main()
