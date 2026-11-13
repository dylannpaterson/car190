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

    def test_central_full_tmm_defaults_match_fit_runners(self):
        self.assertEqual(FULL_TMM_EMCEE_WALKERS, emcee_warmup.EMCEE_WALKERS)
        self.assertEqual(FULL_TMM_EMCEE_STEPS, emcee_warmup.EMCEE_STEPS)
        self.assertEqual(FULL_TMM_EMCEE_SEED, emcee_warmup.EMCEE_SEED)
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

    def test_initialize_records_input_hashes_and_rejects_outputs_outside_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "20261113"
            statistics = self.make_statistics(data_root)
            output = data_root / "production" / "pipeline_runs" / "relative"
            run = pipeline.PipelineRun(statistics, output, RELATIVE_RESPONSE)
            self.assertTrue(run.initialize())
            self.assertEqual(len(run.manifest["input_fingerprints"]), 19)
            self.assertTrue((output / "pipeline_manifest.json").is_file())
            self.assertEqual(run.paths["logs"], output / "logs")

            outside = Path(temporary) / "elsewhere"
            with self.assertRaisesRegex(ValueError, "inside the input data folder"):
                pipeline.PipelineRun(statistics, outside, RELATIVE_RESPONSE).initialize()

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
