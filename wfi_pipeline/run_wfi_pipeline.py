"""Path-driven orchestration for the local WFI fit-to-product pipeline.

Nexus reduction is a separate handoff. Use the relative-response or full-TMM
wrapper, each of which requires only an input path and an output path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from wfi_pipeline_defaults import (
    DETECTORS,
    ERIC_V3_BASE_MODEL,
    FINAL_H5_NAME,
    FULL_TMM_EMCEE_SEED,
    FULL_TMM_EMCEE_STEPS,
    FULL_TMM_EMCEE_WALKERS,
    FULL_TMM_MAP_FTOL,
    FULL_TMM_MAP_MAX_EVALUATIONS,
    FULL_TMM_MAP_XTOL,
    FULL_TMM,
    JOINT_EXPERIMENT_NAME,
    JOINT_MODULATED_BUNDLE_NAME,
    JOINT_FIT_FTOL,
    JOINT_FIT_GTOL,
    JOINT_FIT_MAXITER,
    PRODUCTION_RATE_PRIOR_MEAN_NM_DAY,
    PRODUCTION_SED_WEIGHTING,
    PIPELINE_VERSION,
    RELATIVE_RESPONSE,
    SPECTRAL_CACHE_MAX_THICKNESS_NM,
    SPECTRAL_CACHE_RANK,
    SPECTRAL_CACHE_THICKNESS_STEP_NM,
    SPECTRAL_CACHE_WAVELENGTH_MAX_NM,
    SPECTRAL_CACHE_WAVELENGTH_MIN_NM,
    SPECTRAL_CACHE_WAVELENGTH_STEP_NM,
    THINNING_SPACING_HOURS,
    TILE_SIZE_PIXELS,
    TMM_REFLECTANCE_MODE_COUNT,
    TMM_REFLECTANCE_WAVELENGTH_MAX_NM,
    TMM_REFLECTANCE_WAVELENGTH_MIN_NM,
    TMM_REFLECTANCE_WAVELENGTH_STEP_NM,
    detector_fit_output_root,
    detector_six_led_cache_root,
    detector_spectral_cache_root,
    output_paths,
    pinned_relative_response_cache,
    profile_for,
)
from reset_events import DEFAULT_RESET_CONFIG, load_reset_config


ROOT = Path(__file__).resolve().parents[1]
JOINT_FIT_NAME = JOINT_EXPERIMENT_NAME


def _corrected_stats_file(directory, detector):
    return Path(directory) / (
        f"{detector.lower()}_tile_log_statistics_{TILE_SIZE_PIXELS}_count_corrected.csv"
    )


def _is_complete_statistics_dir(directory):
    directory = Path(directory)
    return directory.is_dir() and all(
        _corrected_stats_file(directory, detector).is_file()
        for detector in DETECTORS
    )


def resolve_input_statistics(input_path):
    """Resolve either a Nexus run root or a corrected-statistics directory."""
    source = Path(input_path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {source}")
    candidates = (
        source / "corrected_statistics_4h_thinned",
        source / "corrected_statistics",
        source,
    )
    for candidate in candidates:
        if _is_complete_statistics_dir(candidate):
            manifest_path = candidate / "program1063_cadence_thinning_manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    policy = manifest.get("policy", {})
                    spacing_hours = float(policy.get("spacing_hours", -1))
                    tile_size = int(policy.get("tile_size", -1))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid cadence-thinning manifest: {manifest_path}"
                    ) from error
                if (
                    spacing_hours != THINNING_SPACING_HOURS
                    or tile_size != TILE_SIZE_PIXELS
                ):
                    raise ValueError(
                        f"Statistics manifest specifies {spacing_hours:g}-hour, "
                        f"{tile_size}-pixel thinning; expected "
                        f"{THINNING_SPACING_HOURS:g} hours and "
                        f"{TILE_SIZE_PIXELS}-pixel tiles: {manifest_path}"
                    )
                return candidate, "already_thinned"
            if candidate.name.endswith("_4h_thinned"):
                return candidate, "already_thinned"
            return candidate, "needs_thinning"
    raise FileNotFoundError(
        f"Expected all 18 corrected 256-pixel statistics files below {source}; "
        "provide a Nexus run root, corrected_statistics/, or a 4-hour-thinned directory"
    )


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistics_input_record(directory):
    directory = Path(directory)
    files = [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in sorted(
            _corrected_stats_file(directory, detector) for detector in DETECTORS
        )
    ]
    manifest = directory / "program1063_cadence_thinning_manifest.json"
    if manifest.is_file():
        files.append({"path": str(manifest.resolve()), "sha256": _sha256(manifest)})
    return files


class PipelineRun:
    def __init__(self, input_path, output_path, route):
        self.input_path = Path(input_path).expanduser().resolve()
        self.output_path = Path(output_path).expanduser().resolve()
        self.profile = profile_for(route)
        self.reset_config = load_reset_config(DEFAULT_RESET_CONFIG)
        self.reset_events = tuple(self.reset_config["events"])
        self.third_decon_day = (
            float(self.reset_events[1]["day"])
            if len(self.reset_events) > 1
            else None
        )
        self.reset_warmup_events = tuple(
            event for event in self.reset_events if event.get("relative_response_warmup")
        )
        self.reset_warmup_start_nm = (
            float(self.reset_warmup_events[0]["relative_response_warmup"]["start_nm"])
            if self.reset_warmup_events
            else None
        )
        self.reset_warmup_log_sigma = (
            float(self.reset_warmup_events[0]["relative_response_warmup"]["log_sigma"])
            if self.reset_warmup_events
            else None
        )
        self.reset_warmup_stages = (
            "third-decon"
            if self.reset_warmup_events
            and self.reset_warmup_events[0]["name"] == "third_decon"
            else (
                f"reset-event:{self.reset_warmup_events[0]['name']}"
                if self.reset_warmup_events
                else None
            )
        )
        warmup_labels = [
            f"{event['name']}_start{event['relative_response_warmup']['start_nm']:g}"
            for event in self.reset_warmup_events
        ]
        if len(warmup_labels) == 1 and self.reset_warmup_events[0]["name"] == "third_decon":
            warmup_labels[0] = f"start{self.reset_warmup_start_nm:g}"
        self.warmup_output_name = (
            "relative_response_constant_rate_warmup_" + "_".join(warmup_labels)
            if warmup_labels
            else "relative_response_constant_rate_warmup"
        )
        self.paths = output_paths(self.output_path, route)
        self.manifest_path = self.paths["manifest"]
        self.manifest = None
        self.statistics_dir = None
        self.thinning_status = None
        self.input_fingerprints = None

    def initialize(self):
        if not self.input_path.is_dir():
            raise FileNotFoundError(f"Input path is not a directory: {self.input_path}")
        self.statistics_dir, self.thinning_status = resolve_input_statistics(
            self.input_path
        )
        try:
            self.output_path.relative_to(self.statistics_dir)
        except ValueError:
            pass
        else:
            raise ValueError(
                "Output path cannot be inside the corrected-statistics directory; "
                "choose a separate model-output directory"
            )
        self.input_fingerprints = _statistics_input_record(self.statistics_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.is_file():
            previous = json.loads(self.manifest_path.read_text())
            if (
                previous.get("input_path") != str(self.input_path)
                or previous.get("model_route") != self.profile.name
                or previous.get("input_fingerprints") != self.input_fingerprints
                or previous.get("pipeline_version") != PIPELINE_VERSION
                or previous.get("settings") != self._settings_record()
            ):
                raise ValueError(
                    "The output path already contains a run for different inputs, "
                    "changed statistics/settings, a different model route, or an "
                    "incompatible pipeline version; "
                    "choose a new --output-path"
                )
            if previous.get("status") == "complete":
                final_h5 = Path(previous.get("final_h5", ""))
                expected_final_h5 = self.paths["h5_root"] / FINAL_H5_NAME
                if (
                    final_h5.resolve() == expected_final_h5.resolve()
                    and final_h5.is_file()
                    and final_h5.with_suffix(".validation.json").is_file()
                ):
                    had_failure = "error" in previous or "failed_utc" in previous
                    previous.pop("error", None)
                    previous.pop("failed_utc", None)
                    if had_failure:
                        _atomic_write_json(self.manifest_path, previous)
                    print(f"Pipeline already complete: {self.manifest_path}", flush=True)
                    return False
            self.manifest = previous
            self.manifest["status"] = "running"
            self.manifest["resumed_utc"] = datetime.now(timezone.utc).isoformat()
        else:
            existing = [path for path in self.output_path.iterdir()]
            if existing:
                raise FileExistsError(
                    f"Output path is not empty and has no pipeline manifest: {self.output_path}"
                )
            self.manifest = {
                "pipeline_version": PIPELINE_VERSION,
                "model_route": self.profile.name,
                "input_path": str(self.input_path),
                "input_fingerprints": self.input_fingerprints,
                "output_path": str(self.output_path),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "settings": self._settings_record(),
                "stages": {},
            }
        self.paths["logs"].mkdir(parents=True, exist_ok=True)
        self.paths["production"].mkdir(parents=True, exist_ok=True)
        self._save_manifest()
        return True

    def _settings_record(self):
        profile = self.profile
        return {
            "tile_size_pixels": profile.tile_size_pixels,
            "thinning_spacing_hours": THINNING_SPACING_HOURS,
            "sed_weighting": PRODUCTION_SED_WEIGHTING,
            "third_decon_day": self.third_decon_day,
            "reset_configuration": {
                "path": str(self.reset_config["path"]),
                "sha256": self.reset_config["sha256"],
                "events": [dict(event) for event in self.reset_events],
            },
            "trajectory_rate_prior_mean_nm_day": PRODUCTION_RATE_PRIOR_MEAN_NM_DAY,
            "workers": profile.workers,
            "six_led_cache_policy": (
                "pinned_input_no_rebuild"
                if profile.use_pinned_six_led_cache
                else "build_from_current_full_tmm_fit"
            ),
            "relative_warmup_third_decon_start_nm": self.reset_warmup_start_nm,
            "relative_warmup_prior_log_sigma": self.reset_warmup_log_sigma,
            "relative_warmup_prior_stages": self.reset_warmup_stages,
            "fit_led_log_offsets": profile.fit_led_log_offsets,
            "led_log_offset_prior_sigma_log": profile.led_log_offset_prior_sigma_log,
            "relative_batch_size": profile.relative_batch_size,
            "spectral_cache": {
                "wavelength_min_nm": SPECTRAL_CACHE_WAVELENGTH_MIN_NM,
                "wavelength_max_nm": SPECTRAL_CACHE_WAVELENGTH_MAX_NM,
                "wavelength_step_nm": SPECTRAL_CACHE_WAVELENGTH_STEP_NM,
                "max_thickness_nm": SPECTRAL_CACHE_MAX_THICKNESS_NM,
                "thickness_step_nm": SPECTRAL_CACHE_THICKNESS_STEP_NM,
                "rank": SPECTRAL_CACHE_RANK,
            },
            "reflectance_modes": {
                "count": TMM_REFLECTANCE_MODE_COUNT,
                "wavelength_min_nm": TMM_REFLECTANCE_WAVELENGTH_MIN_NM,
                "wavelength_max_nm": TMM_REFLECTANCE_WAVELENGTH_MAX_NM,
                "wavelength_step_nm": TMM_REFLECTANCE_WAVELENGTH_STEP_NM,
            },
            "joint_fit_name": JOINT_FIT_NAME,
            "portable_bundle_filenames": {
                "base": "wfi_all_2nm_rank16.npz",
                "modulated": JOINT_MODULATED_BUNDLE_NAME,
            },
            "h5_filenames": {
                "final_directory": "final",
                "intermediate_directory": "intermediates/h5",
                "final_model": FINAL_H5_NAME,
            },
            "joint_optimizer": {
                "method": "L-BFGS-B",
                "maxiter": JOINT_FIT_MAXITER,
                "ftol": JOINT_FIT_FTOL,
                "gtol": JOINT_FIT_GTOL,
            },
            "full_tmm_initialization": {
                "emcee_walkers": FULL_TMM_EMCEE_WALKERS,
                "emcee_steps": FULL_TMM_EMCEE_STEPS,
                "emcee_seed": FULL_TMM_EMCEE_SEED,
                "complex_q_parameters": False,
                "constant_rate_initialization": True,
                "map_max_evaluations": FULL_TMM_MAP_MAX_EVALUATIONS,
                "map_ftol": FULL_TMM_MAP_FTOL,
                "map_xtol": FULL_TMM_MAP_XTOL,
            },
        }

    def _save_manifest(self):
        _atomic_write_json(self.manifest_path, self.manifest)

    def _run_command(self, stage, command, *, detector=None, outputs=()):
        key = f"{stage}/{detector}" if detector else stage
        previous = self.manifest["stages"].get(key, {})
        command_values = [str(value) for value in command]
        output_values = [str(Path(value).resolve()) for value in outputs]
        if (
            previous.get("status") == "complete"
            and previous.get("command") == command_values
            and previous.get("outputs") == output_values
            and all(Path(item).is_file() for item in outputs)
        ):
            print(f"{key}: already complete; skipping", flush=True)
            return
        log_path = self.paths["logs"] / stage / f"{detector or 'all'}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "status": "running",
            "command": command_values,
            "log": str(log_path.resolve()),
            "outputs": output_values,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.manifest["stages"][key] = record
        self._save_manifest()
        started = time.monotonic()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nCOMMAND: {shlex.join([str(value) for value in command])}\n")
            log.flush()
            result = subprocess.run(
                [str(value) for value in command],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record["return_code"] = int(result.returncode)
        record["seconds"] = float(time.monotonic() - started)
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["status"] = "complete" if result.returncode == 0 else "failed"
        self._save_manifest()
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)

    def _prepare_statistics(self):
        source, thinning_status = self.statistics_dir, self.thinning_status
        if thinning_status == "already_thinned":
            self.manifest["statistics_path"] = str(source)
            self.manifest["statistics_handoff"] = "input_already_4hour_thinned"
            self._save_manifest()
            return source
        output = self.paths["corrected_statistics_4h"]
        if output.is_dir() and _is_complete_statistics_dir(output):
            self.manifest["statistics_path"] = str(output)
            self.manifest["statistics_handoff"] = "reused_pipeline_thinning_output"
            self._save_manifest()
            return output
        command = [
            sys.executable,
                str(ROOT / "wfi_pipeline" / "thin_wfi_statistics.py"),
            "--source-dir", str(source),
            "--output-dir", str(output),
            "--spacing-hours", str(THINNING_SPACING_HOURS),
            "--tile-size", str(TILE_SIZE_PIXELS),
        ]
        expected = [_corrected_stats_file(output, detector) for detector in DETECTORS]
        expected.append(output / "program1063_cadence_thinning_manifest.json")
        self._run_command("thin_statistics", command, outputs=expected)
        if not _is_complete_statistics_dir(output):
            raise RuntimeError(f"Thinning did not produce complete all-detector statistics: {output}")
        self.manifest["statistics_path"] = str(output)
        self.manifest["statistics_handoff"] = "thinned_4hour_tile256"
        self._save_manifest()
        return output

    @staticmethod
    def _tile_archives(root, filename="posterior_samples.npz"):
        return all(
            (Path(root) / f"tile_{tile_id:03d}" / filename).is_file()
            for tile_id in range(256)
        )

    @staticmethod
    def _tile_outputs(root, filename):
        return [
            Path(root) / f"tile_{tile_id:03d}" / filename
            for tile_id in range(256)
        ]

    @staticmethod
    def _hardlink_or_copy(source, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return destination

    def _previous_relative_outputs(self, statistics_dir, detector):
        """Return a compatible prior relative fit tree, if one exists beside the input."""
        if self.profile.name != RELATIVE_RESPONSE:
            return None
        from run_wfi_relative_response_production import cached_fit_matches
        from wfi_relative_response import resolve_portable_path
        from wfi_production_layout import tile_statistics_path

        source_root = Path(statistics_dir) / "production" / detector.lower() / "relative_response"
        mode_path = source_root / "relative_response_mode.json"
        if not mode_path.is_file():
            return None
        try:
            mode = json.loads(mode_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        input_statistics = _corrected_stats_file(statistics_dir, detector)
        source_statistics = resolve_portable_path(mode.get("statistics", ""))
        warmup_root = resolve_portable_path(mode.get("warmup_prior_root", ""))
        cache_root = pinned_relative_response_cache(detector)
        try:
            expected_mode = (
                mode.get("tile_size") == TILE_SIZE_PIXELS
                and mode.get("fit_led_log_offsets") is True
                and float(mode.get("led_log_offset_prior_sigma_log", -1.0))
                == self.profile.led_log_offset_prior_sigma_log
                and mode.get("third_decon_day") == self.third_decon_day
                and mode.get("thickness_prior_log_sigma")
                == self.reset_warmup_log_sigma
                and mode.get("thickness_prior_stages")
                == self.reset_warmup_stages
                and mode.get("reset_config_sha256") == self.reset_config["sha256"]
                and mode.get("batch_size") == self.profile.relative_batch_size
                and mode.get("joint_batch") is False
                and float(mode.get("trajectory_entropy_weight", -1.0)) == 0.0
                and resolve_portable_path(mode.get("response_cache_root", ""))
                == cache_root.resolve()
                and source_statistics.is_file()
                and _sha256(source_statistics) == _sha256(input_statistics)
            )
        except (TypeError, ValueError, OSError):
            return None
        if not expected_mode:
            return None

        warmup_mode_path = warmup_root / "relative_response_constant_rate_warmup_mode.json"
        try:
            warmup_mode = json.loads(warmup_mode_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if (
            warmup_mode.get("detector") != detector
            or warmup_mode.get("tile_size") != TILE_SIZE_PIXELS
            or warmup_mode.get("third_decon_day") != self.third_decon_day
            or warmup_mode.get("third_decon_initial_thickness_start_nm")
            != (
                self.reset_warmup_start_nm
                if self.reset_warmup_events
                and self.reset_warmup_events[0]["name"] == "third_decon"
                else None
            )
            or warmup_mode.get("reset_event_initial_thickness_starts_nm")
            != {
                event["name"]: float(event["relative_response_warmup"]["start_nm"])
                for event in self.reset_warmup_events
            }
            or warmup_mode.get("reset_config_sha256") != self.reset_config["sha256"]
            or warmup_mode.get("fit_led_log_offsets") is not True
            or float(warmup_mode.get("led_log_offset_prior_sigma_log", -1.0))
            != self.profile.led_log_offset_prior_sigma_log
            or resolve_portable_path(warmup_mode.get("statistics", ""))
            != input_statistics.resolve()
            or not self._tile_archives(warmup_root / "fits", "constant_rate_map.npz")
        ):
            return None

        fit_root = source_root / "fits"
        split_statistics_root = source_root / "tile_statistics"
        if not self._tile_archives(fit_root):
            return None
        for tile_id in range(256):
            tile_stats = tile_statistics_path(
                split_statistics_root, detector, TILE_SIZE_PIXELS, tile_id
            )
            tile_fit = fit_root / f"tile_{tile_id:03d}"
            cache_path = cache_root / f"tile_{tile_id:03d}" / "relative_response.npz"
            if (
                not tile_stats.is_file()
                or not cache_path.is_file()
                or not cached_fit_matches(
                    tile_fit,
                    tile_stats,
                    cache_path,
                    third_decon_day=self.third_decon_day,
                    fit_led_log_offsets=self.profile.fit_led_log_offsets,
                    led_log_offset_prior_sigma_log=(
                        self.profile.led_log_offset_prior_sigma_log
                    ),
                    warmup_prior_root=warmup_root,
                    thickness_prior_log_sigma=self.reset_warmup_log_sigma,
                    thickness_prior_stages=self.reset_warmup_stages,
                    reset_events=self.reset_events,
                    reset_event_initial_thickness_starts_nm={
                        event["name"]: event["relative_response_warmup"]["start_nm"]
                        for event in self.reset_warmup_events
                    },
                )
            ):
                return None

        joint_root = source_root / "experiments" / JOINT_FIT_NAME
        joint_summary = joint_root / "summary.json"
        joint_archive_paths = (
            joint_root / "joint_fit.npz",
            joint_root / "joint_weekly_ice_fit.npz",
            joint_root / "training_inputs.npz",
        )
        if joint_summary.is_file() and all(path.is_file() for path in joint_archive_paths):
            try:
                summary = json.loads(joint_summary.read_text())
            except (OSError, json.JSONDecodeError):
                summary = {}
            if not (
                summary.get("detector") == detector
                and summary.get("optimizer_success") is True
                and summary.get("response_convention") == "eric-reader"
                and summary.get("ice_histories_refitted") is True
                and summary.get("ice_rate_control_spacing_days") == 2.0
                and resolve_portable_path(summary.get("ice_history_fit_root", ""))
                == fit_root.resolve()
                and resolve_portable_path(summary.get("optical_cache_root", ""))
                == cache_root.resolve()
            ):
                joint_root = None
        else:
            joint_root = None

        return {
            "source_fit_root": source_root,
            "source_warmup_root": warmup_root,
            "source_joint_root": joint_root,
        }

    def _reuse_previous_relative_outputs(self, statistics_dir):
        """Seed this run from matching input-adjacent fits without refitting them."""
        if self.profile.name != RELATIVE_RESPONSE:
            return
        reused = {}
        for detector in DETECTORS:
            fit_stage = f"fit-check/{detector}"
            if self.manifest["stages"].get(fit_stage, {}).get("status") == "complete":
                continue
            prior = self._previous_relative_outputs(statistics_dir, detector)
            if prior is None:
                continue

            source_root = prior["source_fit_root"]
            target_root = detector_fit_output_root(
                self.output_path, self.profile.name, detector
            )
            target_root.parent.mkdir(parents=True, exist_ok=True)
            if not target_root.exists():
                shutil.copytree(
                    source_root,
                    target_root,
                    copy_function=self._hardlink_or_copy,
                    ignore=shutil.ignore_patterns("experiments", "*.png", "*.pdf"),
                )
            else:
                for directory in ("fits", "tile_statistics"):
                    if not (target_root / directory).is_dir():
                        shutil.copytree(
                            source_root / directory,
                            target_root / directory,
                            copy_function=self._hardlink_or_copy,
                        )
                for filename in ("relative_response_mode.json", "tile_rate_posterior.csv"):
                    source_file = source_root / filename
                    target_file = target_root / filename
                    if source_file.is_file() and not target_file.exists():
                        self._hardlink_or_copy(source_file, target_file)

            source_warmup = prior["source_warmup_root"]
            target_warmup = (
                self.paths["production"] / detector.lower()
                / self.warmup_output_name
            )
            target_warmup.parent.mkdir(parents=True, exist_ok=True)
            if not target_warmup.exists():
                shutil.copytree(
                    source_warmup,
                    target_warmup,
                    copy_function=self._hardlink_or_copy,
                )

            fit_outputs = [
                target_root / "relative_response_mode.json",
                target_root / "tile_rate_posterior.csv",
                *self._tile_outputs(target_root / "fits", "posterior_samples.npz"),
            ]
            warmup_outputs = [
                target_warmup / "relative_response_constant_rate_warmup_mode.json",
                *self._tile_outputs(target_warmup / "fits", "constant_rate_map.npz"),
            ]
            if not all(path.is_file() for path in fit_outputs + warmup_outputs):
                continue

            self.manifest["stages"][f"relative_constant_rate_warmup/{detector}"] = {
                "status": "complete",
                "outputs": [str(path.resolve()) for path in warmup_outputs],
                "reused_from": str(source_warmup.resolve()),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.manifest["stages"][f"relative_response_fit/{detector}"] = {
                "status": "complete",
                "outputs": [str(path.resolve()) for path in fit_outputs],
                "reused_from": str(source_root.resolve()),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            self.manifest["stages"][fit_stage] = {
                "status": "complete",
                "fit_root": str(target_root.resolve()),
                "tile_archive_count": 256,
                "reused_from": str(source_root.resolve()),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }

            if prior["source_joint_root"] is not None:
                target_joint = target_root / "experiments" / JOINT_FIT_NAME
                target_joint.parent.mkdir(parents=True, exist_ok=True)
                if not target_joint.exists():
                    shutil.copytree(
                        prior["source_joint_root"],
                        target_joint,
                        copy_function=self._hardlink_or_copy,
                    )
                reused[detector] = str(prior["source_joint_root"].resolve())

            self._save_manifest()

        if reused:
            self.manifest["reused_joint_fit_sources"] = reused
            self._save_manifest()

    @staticmethod
    def _cache_complete(root, filename):
        return all(
            (Path(root) / f"tile_{tile_id:03d}" / filename).is_file()
            for tile_id in range(256)
        )

    def _fit_detector(self, statistics_dir, detector):
        profile = self.profile
        fit_root = detector_fit_output_root(self.output_path, profile.name, detector)
        if profile.name == FULL_TMM:
            command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / "run_wfi_production.py"),
                "--statistics-dir", str(statistics_dir),
                "--detector", detector,
                "--tile-size", str(TILE_SIZE_PIXELS),
                "--output-root", str(fit_root),
                "--reset-config", str(self.reset_config["path"]),
                "--workers", str(profile.workers),
            ]
            status = fit_root / "production_status.json"
            self._run_command(
                "full_tmm_fit", command, detector=detector,
                outputs=[
                    status,
                    *self._tile_outputs(fit_root / "fits", "posterior_samples.npz"),
                ],
            )
        else:
            cache_root = pinned_relative_response_cache(detector)
            if not self._cache_complete(cache_root, "relative_response.npz"):
                raise FileNotFoundError(f"Pinned six-LED response cache is missing: {cache_root}")
            warmup_root = (
                self.paths["production"] / detector.lower()
                / self.warmup_output_name
            )
            warmup_outputs = [
                warmup_root / "relative_response_constant_rate_warmup_mode.json",
                *self._tile_outputs(warmup_root / "fits", "constant_rate_map.npz"),
            ]
            warmup_command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / "run_wfi_relative_response_constant_rate_warmup.py"),
                "--statistics-dir", str(statistics_dir),
                "--detector", detector,
                "--tile-size", str(TILE_SIZE_PIXELS),
                "--response-cache-root", str(cache_root),
                "--output-root", str(warmup_root),
                "--reset-config", str(self.reset_config["path"]),
                "--fit-led-log-offsets",
                "--led-log-offset-prior-sigma-log",
                str(profile.led_log_offset_prior_sigma_log),
                "--allow-external-cache-source",
                "--workers", str(profile.workers),
            ]
            if not all(path.is_file() for path in warmup_outputs):
                self._run_command(
                    "relative_constant_rate_warmup", warmup_command,
                    detector=detector, outputs=warmup_outputs,
                )
            fit_outputs = [
                fit_root / "relative_response_mode.json",
                fit_root / "tile_rate_posterior.csv",
                *self._tile_outputs(fit_root / "fits", "posterior_samples.npz"),
            ]
            fit_command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / "run_wfi_production.py"),
                "--statistics-dir", str(statistics_dir),
                "--detector", detector,
                "--tile-size", str(TILE_SIZE_PIXELS),
                "--output-root", str(fit_root),
                "--response-cache-root", str(cache_root),
                "--reset-config", str(self.reset_config["path"]),
                "--warmup-prior-root", str(warmup_root),
                "--thickness-prior-log-sigma", str(self.reset_warmup_log_sigma),
                "--thickness-prior-stages", str(self.reset_warmup_stages),
                "--fit-led-log-offsets",
                "--led-log-offset-prior-sigma-log",
                str(profile.led_log_offset_prior_sigma_log),
                "--batch-size", str(profile.relative_batch_size),
                "--allow-external-cache-source",
                "--workers", str(profile.workers),
            ]
            self._run_command(
                "relative_response_fit", fit_command, detector=detector,
                outputs=fit_outputs,
            )

        if not self._tile_archives(fit_root / "fits"):
            raise RuntimeError(f"{detector}: model fit did not produce 256 tile archives")
        return fit_root

    def _fit_all_detectors(self, statistics_dir):
        fit_roots = {}
        for detector in DETECTORS:
            fit_root = detector_fit_output_root(
                self.output_path, self.profile.name, detector
            )
            fit_root.parent.mkdir(parents=True, exist_ok=True)
            key = f"fit-check/{detector}"
            previous = self.manifest["stages"].get(key, {})
            if previous.get("status") == "complete" and self._tile_archives(fit_root / "fits"):
                fit_roots[detector] = fit_root
                continue
            fit_root = self._fit_detector(statistics_dir, detector)
            self.manifest["stages"][key] = {
                "status": "complete",
                "fit_root": str(fit_root.resolve()),
                "tile_archive_count": 256,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._save_manifest()
            fit_roots[detector] = fit_root
        return fit_roots

    def _build_full_tmm_six_led_cache(self, statistics_dir, detector, fit_root):
        cache_root = detector_six_led_cache_root(
            self.output_path, self.profile.name, detector
        )
        if self._cache_complete(cache_root, "relative_response.npz"):
            return cache_root
        command = [
            sys.executable,
            str(ROOT / "wfi_pipeline" / "build_wfi_relative_response_cache.py"),
            "--statistics-root", str(fit_root / "tile_statistics"),
            "--fit-root", str(fit_root / "fits"),
            "--output-root", str(cache_root),
            "--detector", detector,
            "--tile-size", str(TILE_SIZE_PIXELS),
        ]
        self._run_command(
            "full_tmm_six_led_cache", command, detector=detector,
            outputs=self._tile_outputs(cache_root, "relative_response.npz"),
        )
        if not self._cache_complete(cache_root, "relative_response.npz"):
            raise RuntimeError(f"{detector}: six-LED cache is incomplete: {cache_root}")
        return cache_root

    def _spectral_cache_fit_root(self, fit_root, six_led_cache_root):
        """Return the TMM optical-fit tree used for spectral response grids."""
        if self.profile.build_six_led_cache_from_current_tmm:
            return Path(fit_root) / "fits"

        from run_wfi_relative_response_production import cache_source_paths

        source_caches = sorted(
            Path(six_led_cache_root).glob("tile_*/relative_response.npz")
        )
        if not source_caches:
            raise FileNotFoundError(
                f"No six-LED source caches found under {six_led_cache_root}"
            )
        source_fit, _ = cache_source_paths(source_caches[0])
        return source_fit.parent.parent

    def _build_spectral_cache(self, detector, fit_root, six_led_cache_root):
        cache_root = detector_spectral_cache_root(
            self.output_path, self.profile.name, detector
        )
        if self._cache_complete(cache_root, "spectral_response.npz"):
            return cache_root
        spectral_fit_root = self._spectral_cache_fit_root(
            fit_root, six_led_cache_root
        )
        command = [
            sys.executable,
            str(ROOT / "wfi_pipeline" / "build_wfi_wavelength_response_cache.py"),
            "--fit-root", str(spectral_fit_root),
            "--output-root", str(cache_root),
            "--detector", detector,
            "--skip-existing",
            "--wavelength-min-nm", str(SPECTRAL_CACHE_WAVELENGTH_MIN_NM),
            "--wavelength-max-nm", str(SPECTRAL_CACHE_WAVELENGTH_MAX_NM),
            "--wavelength-step-nm", str(SPECTRAL_CACHE_WAVELENGTH_STEP_NM),
            "--max-thickness-nm", str(SPECTRAL_CACHE_MAX_THICKNESS_NM),
            "--thickness-step-nm", str(SPECTRAL_CACHE_THICKNESS_STEP_NM),
            "--rank", str(SPECTRAL_CACHE_RANK),
        ]
        self._run_command(
            "spectral_response_cache", command, detector=detector,
            outputs=self._tile_outputs(cache_root, "spectral_response.npz"),
        )
        if not self._cache_complete(cache_root, "spectral_response.npz"):
            raise RuntimeError(f"{detector}: spectral-response cache is incomplete: {cache_root}")
        return cache_root

    def _build_base_bundle(self):
        from wfi_wavelength_response_bundle import load_bundle

        fit_subdir = "relative_response" if self.profile.name == RELATIVE_RESPONSE else "full_tmm"
        cache_root = self.paths["spectral_cache_root"]
        output = self.paths["bundle_root"] / "wfi_all_2nm_rank16.npz"
        if output.is_file():
            bundle = load_bundle(output)
            detector_values = bundle["detectors"].astype(str)
            tile_counts = {
                detector: int((detector_values == detector).sum())
                for detector in DETECTORS
            }
            if len(bundle["tile_ids"]) == 4608 and all(
                count == 256 for count in tile_counts.values()
            ):
                self.manifest["stages"]["base_full_wavelength_bundle"] = {
                    "status": "complete",
                    "outputs": [str(output.resolve())],
                    "reused_verified_output": True,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                }
                self._save_manifest()
                return output
            raise ValueError(f"Existing base bundle does not match this pipeline run: {output}")
        command = [
            sys.executable,
            str(ROOT / "wfi_pipeline" / "build_wfi_wavelength_response_bundle.py"),
            "--cache-root", str(cache_root),
            "--fit-root", str(self.paths["production"]),
            "--fit-subdir", fit_subdir,
            "--output", str(output),
            "--tile-size", str(TILE_SIZE_PIXELS),
        ]
        command.extend(item for detector in DETECTORS for item in ("--detector", detector))
        self._run_command("base_full_wavelength_bundle", command, outputs=[output])
        bundle = load_bundle(output)
        detector_values = bundle["detectors"].astype(str)
        tile_counts = {
            detector: int((detector_values == detector).sum())
            for detector in DETECTORS
        }
        if len(bundle["tile_ids"]) != 4608 or any(
            count != 256 for count in tile_counts.values()
        ):
            raise RuntimeError(f"Base bundle is missing detector/tile coverage: {output}")
        return output

    def _run_joint_fits(self, statistics_dir, fit_roots, six_led_roots):
        joint_name = JOINT_FIT_NAME
        for detector in DETECTORS:
            fit_root = fit_roots[detector]
            optical_root = six_led_roots[detector]
            joint_output = (
                self.paths["production"] / detector.lower() / self.profile.name
                / "experiments" / joint_name
            )
            summary_path = joint_output / "summary.json"
            joint_outputs = [
                summary_path,
                joint_output / "joint_fit.npz",
                joint_output / "joint_weekly_ice_fit.npz",
            ]
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text())
                    if (
                        summary.get("optimizer_success")
                        and summary.get("response_convention") == "eric-reader"
                        and all(path.is_file() for path in joint_outputs)
                    ):
                        self.manifest["stages"][f"joint_modulated_fit/{detector}"] = {
                            "status": "complete",
                            "outputs": [str(path.resolve()) for path in joint_outputs],
                            "finished_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        self._save_manifest()
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / "run_joint_car190_amber_inverse_wavelength_maxent.py"),
                "--detector", detector,
                "--statistics-dir", str(statistics_dir),
                "--reset-config", str(self.reset_config["path"]),
                "--production-root", str(self.paths["production"]),
                "--joint-input-root", str(fit_root),
                "--optical-cache-root", str(optical_root),
                "--response-convention", "eric-reader",
                "--output-dir", str(joint_output),
                "--maxiter", str(JOINT_FIT_MAXITER),
                "--ftol", str(JOINT_FIT_FTOL),
                "--gtol", str(JOINT_FIT_GTOL),
            ]
            checkpoint = joint_output / "optimizer_checkpoint.npz"
            if checkpoint.is_file():
                command.append("--resume")
            self._run_command(
                "joint_modulated_fit", command, detector=detector,
                outputs=joint_outputs,
            )
            summary = json.loads(summary_path.read_text())
            if not summary.get("optimizer_success"):
                raise RuntimeError(f"{detector}: joint-modulated fit did not converge")

    def _run_bundle_and_h5(self, base_bundle):
        from wfi_wavelength_response_bundle import load_bundle

        detector_args = [item for detector in DETECTORS for item in ("--detector", detector)]
        fit_subdir = "relative_response" if self.profile.name == RELATIVE_RESPONSE else "full_tmm"
        bundle_root = self.paths["bundle_root"]
        bundle_root.mkdir(parents=True, exist_ok=True)
        modulated_bundle = bundle_root / JOINT_MODULATED_BUNDLE_NAME
        checkpoint_dir = modulated_bundle.with_name(
            modulated_bundle.name + ".tile_checkpoints"
        )
        checkpoint_manifest = checkpoint_dir / "build_manifest.json"
        bundle_is_complete = False
        if modulated_bundle.is_file():
            try:
                existing_bundle = load_bundle(modulated_bundle)
            except Exception as error:
                if not checkpoint_manifest.is_file():
                    raise RuntimeError(
                        f"Existing modulated bundle is unreadable and has no "
                        f"compatible checkpoints: {modulated_bundle}"
                    ) from error
                self.manifest["stages"]["joint_modulated_bundle"] = {
                    "status": "resume_required",
                    "error": f"{type(error).__name__}: {error}",
                }
                self._save_manifest()
            else:
                detector_values = existing_bundle["detectors"].astype(str)
                tile_counts = {
                    detector: int((detector_values == detector).sum())
                    for detector in DETECTORS
                }
                if len(existing_bundle["tile_ids"]) != 4608 or any(
                    count != 256 for count in tile_counts.values()
                ):
                    raise ValueError(
                        f"Existing modulated bundle has incomplete detector/tile "
                        f"coverage: {modulated_bundle}"
                    )
                bundle_is_complete = True
                self.manifest["stages"]["joint_modulated_bundle"] = {
                    "status": "complete",
                    "outputs": [str(modulated_bundle.resolve())],
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "reused_verified_output": True,
                }
                self._save_manifest()
        if not bundle_is_complete:
            command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / "build_modulated_wfi_wavelength_response_bundle.py"),
                "--base-bundle", str(base_bundle),
                "--fit-root", str(self.paths["production"]),
                "--fit-subdir", fit_subdir,
                "--modulation-root", str(self.paths["production"]),
                "--modulation-fit", "joint-car190-amber-2day",
                "--joint-fit-name", JOINT_FIT_NAME,
                "--joint-fit-subdir", self.profile.name,
                "--output", str(modulated_bundle),
                "--checkpoint-interval", "128",
                *detector_args,
            ]
            if checkpoint_manifest.is_file():
                command.append("--resume")
            self._run_command("joint_modulated_bundle", command, outputs=[modulated_bundle])
            completed_bundle = load_bundle(modulated_bundle)
            detector_values = completed_bundle["detectors"].astype(str)
            tile_counts = {
                detector: int((detector_values == detector).sum())
                for detector in DETECTORS
            }
            if len(completed_bundle["tile_ids"]) != 4608 or any(
                count != 256 for count in tile_counts.values()
            ):
                raise RuntimeError(
                    f"Joint-modulated bundle is missing detector/tile coverage: "
                    f"{modulated_bundle}"
                )

        intermediate_h5_root = self.paths["h5_intermediate_root"]
        intermediate_h5_root.mkdir(parents=True, exist_ok=True)
        reflectance_modes = intermediate_h5_root / "tmm_joint_modulated_reflectance_modes.h5"
        if not reflectance_modes.is_file():
            command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / "build_tmm_complex_reflectance_modes.py"),
                "--bundle", str(modulated_bundle),
                "--variant", "joint_modulated",
                "--mode-count", str(TMM_REFLECTANCE_MODE_COUNT),
                "--wavelength-min-nm", str(TMM_REFLECTANCE_WAVELENGTH_MIN_NM),
                "--wavelength-max-nm", str(TMM_REFLECTANCE_WAVELENGTH_MAX_NM),
                "--wavelength-step-nm", str(TMM_REFLECTANCE_WAVELENGTH_STEP_NM),
                "--output", str(reflectance_modes),
                *detector_args,
            ]
            self._run_command("joint_reflectance_modes", command, outputs=[reflectance_modes])

        history_reference = intermediate_h5_root / "joint_ice_history_reference.h5"
        if not history_reference.is_file():
            command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / "export_joint_ice_history_reference.py"),
                "--bundle", str(modulated_bundle),
                "--output", str(history_reference),
            ]
            self._run_command("joint_history_reference", command, outputs=[history_reference])

        base_v3 = ERIC_V3_BASE_MODEL
        if not base_v3.is_file():
            raise FileNotFoundError(f"Pinned Eric v3 dataset contract is missing: {base_v3}")
        prototype = intermediate_h5_root / "ice_response_model_prototype.h5"
        if not prototype.is_file():
            command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / "assemble_eric_v3_ice_response_candidate.py"),
                "--base-v3-h5", str(base_v3),
                "--reflectance-modes", str(reflectance_modes),
                "--ice-history-reference", str(history_reference),
                "--output", str(prototype),
            ]
            self._run_command("assemble_h5_prototype", command, outputs=[prototype])

        final_h5 = self.paths["h5_root"] / FINAL_H5_NAME
        validation = final_h5.with_suffix(".validation.json")
        if not final_h5.is_file() or not validation.is_file():
            command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / "package_joint_h5_unchanged_contract.py"),
                "--source", str(prototype),
                "--output", str(final_h5),
            ]
            if final_h5.exists():
                command.append("--force")
            self._run_command("package_final_h5", command, outputs=[final_h5, validation])

        for stage in (
            "joint_reflectance_modes",
            "joint_history_reference",
            "assemble_h5_prototype",
        ):
            record = self.manifest["stages"].get(stage, {})
            if record.get("status") == "complete":
                record["consumed_by_final_h5"] = str(final_h5.resolve())
                record["output_role"] = "intermediate_h5"
                record["retained_for_pipeline_audit"] = True
                self.manifest["stages"][stage] = record
        self._save_manifest()
        return modulated_bundle, final_h5

    def _run_plots(self, final_h5):
        comparison_dir = self.paths["plots"] / "car201"
        comparison_command = [
            sys.executable,
            str(ROOT / "wfi_pipeline" / "compare_eric_v3_amber_car201.py"),
            "--statistics-dir", str(self.output_path),
            "--joint-fit-name", f"{self.profile.name}/experiments/{JOINT_FIT_NAME}",
            "--include-fit-reference",
            "--output-dir", str(comparison_dir),
            "--joint-modulated-model", str(final_h5),
        ]
        comparison_command.extend(item for detector in DETECTORS for item in ("--detector", detector))
        self._run_command(
            "joint_car201_diagnostics", comparison_command,
            outputs=[comparison_dir / "car201_comparison.pdf",
                     comparison_dir / "amber_eric_v3_vs_joint_modulated_measurement_predictions.csv"],
        )
        predictions = comparison_dir / "amber_eric_v3_vs_joint_modulated_measurement_predictions.csv"
        for script in ("plot_joint_h5_ice_correction_epoch_filter.py",
                       "plot_joint_h5_car201_error_metrics_epoch_filter.py"):
            command = [
                sys.executable,
                str(ROOT / "wfi_pipeline" / script),
                "--predictions", str(predictions),
                "--output-dir", str(comparison_dir),
            ]
            self._run_command(
                "joint_epoch_filter_diagnostics", command,
                outputs=[comparison_dir / "ice_correction_by_filter_epoch.png"]
                if script.startswith("plot_joint_h5_ice") else
                [comparison_dir / "car201_rmse_by_filter_epoch.png",
                 comparison_dir / "car201_nmad_by_filter_epoch.png"],
            )

    def run(self):
        if not self.initialize():
            return
        try:
            statistics_dir = self._prepare_statistics()
            self.manifest["statistics_path"] = str(statistics_dir.resolve())
            self._save_manifest()
            self._reuse_previous_relative_outputs(statistics_dir)

            fit_roots = self._fit_all_detectors(statistics_dir)
            six_led_roots = {}
            for detector in DETECTORS:
                fit_root = fit_roots[detector]
                if self.profile.build_six_led_cache_from_current_tmm:
                    six_led_roots[detector] = self._build_full_tmm_six_led_cache(
                        statistics_dir, detector, fit_roots[detector]
                    )
                else:
                    six_led_roots[detector] = detector_six_led_cache_root(
                        self.output_path, self.profile.name, detector
                    )
                    if not self._cache_complete(
                        six_led_roots[detector], "relative_response.npz"
                    ):
                        raise FileNotFoundError(six_led_roots[detector])
                self._build_spectral_cache(
                    detector, fit_roots[detector], six_led_roots[detector]
                )
                self._run_tmm_plots(statistics_dir, detector, fit_roots[detector])

            self._run_global_rate_plots(statistics_dir)
            base_bundle = self._build_base_bundle()
            self._run_joint_fits(statistics_dir, fit_roots, six_led_roots)
            modulated_bundle, final_h5 = self._run_bundle_and_h5(base_bundle)
            self._run_plots(final_h5)
            artifact_paths = {
                "base_bundle": base_bundle,
                "modulated_bundle": modulated_bundle,
                "final_h5": final_h5,
                "final_h5_validation": final_h5.with_suffix(".validation.json"),
            }
            self.manifest["output_artifacts"] = {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in artifact_paths.items()
            }
            self.manifest["detector_products"] = {
                detector: {
                    "fit_root": str(fit_roots[detector].resolve()),
                    "six_led_cache_root": str(six_led_roots[detector].resolve()),
                    "spectral_cache_root": str(
                        detector_spectral_cache_root(
                            self.output_path, self.profile.name, detector
                        ).resolve()
                    ),
                    "joint_fit_root": str(
                        (
                            self.paths["production"]
                            / detector.lower()
                            / self.profile.name
                            / "experiments"
                            / JOINT_FIT_NAME
                        ).resolve()
                    ),
                }
                for detector in DETECTORS
            }
            self.manifest["base_bundle"] = str(base_bundle.resolve())
            self.manifest["modulated_bundle"] = str(modulated_bundle.resolve())
            self.manifest["final_h5"] = str(final_h5.resolve())
            self.manifest["status"] = "complete"
            self.manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
            self.manifest.pop("error", None)
            self.manifest.pop("failed_utc", None)
            self._save_manifest()
            print(f"Pipeline complete: {final_h5}", flush=True)
        except BaseException as error:
            self.manifest["status"] = "failed"
            self.manifest["error"] = f"{type(error).__name__}: {error}"
            self.manifest["failed_utc"] = datetime.now(timezone.utc).isoformat()
            self._save_manifest()
            raise

    def _run_tmm_plots(self, statistics_dir, detector, fit_root):
        plot_outputs = [
            fit_root / name
            for name in (
                "numpyro_ice_trajectory.png",
                "numpyro_rate_histogram_pre_post_reset.png",
                "numpyro_np_trajectory.png",
                "numpyro_tile_ar_maps.png",
                f"numpyro_tile_rate_maps_{detector}.png",
                "numpyro_tile_fit.png",
                "numpyro_tile_data_model_vs_time.png",
                "numpyro_tile_fit_scatter.png",
                "numpyro_map_optimization.png",
            )
        ]
        command = [
            sys.executable,
            str(ROOT / "wfi_pipeline" / "plot_wfi_independent_production.py"),
            "--root", str(statistics_dir),
            "--detector", detector,
            "--fit-root", str(fit_root),
            "--statistics-path", str(_corrected_stats_file(statistics_dir, detector)),
            "--tile-size", str(TILE_SIZE_PIXELS),
        ]
        self._run_command(
            "tmm_diagnostics", command, detector=detector,
            outputs=plot_outputs,
        )

    def _run_global_rate_plots(self, statistics_dir):
        plot_root = self.paths["plots"]
        focal_plane = plot_root / "focal_plane"
        trajectories = plot_root / "trajectories"
        recent_trajectories = plot_root / "trajectories_last_7d"
        outputs = [
            focal_plane / "wfi_full_focal_plane_rate_maps.png",
            focal_plane / "wfi_full_focal_plane_aom_rate_jump_percent.png",
            trajectories / "wfi_all_detector_ice_rate_trajectories.png",
            recent_trajectories
            / "wfi_all_detector_ice_rate_trajectories_last_7d.png",
        ]
        command = [
            sys.executable,
            str(ROOT / "wfi_pipeline" / "run_wfi_global_rate_plots.py"),
            "--statistics-dir", str(statistics_dir),
            "--production-root", str(self.paths["production"]),
            "--route", str(self.profile.name),
            "--output-dir", str(plot_root),
        ]
        self._run_command(
            "global_rate_diagnostics", command, outputs=outputs
        )


def main_for_route(route, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Run the WFI {route.replace('_', ' ')} fit-to-product pipeline."
    )
    parser.add_argument("--input-path", type=Path, required=True,
                        help="Nexus-transferred corrected statistics or its dated run root.")
    parser.add_argument("--output-path", type=Path, required=True,
                        help="Dedicated local pipeline output directory inside the data folder.")
    args = parser.parse_args(argv)
    PipelineRun(args.input_path, args.output_path, route).run()
