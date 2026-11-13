# Archived exploratory scripts

These scripts were standalone diagnostics, one-off fits, or parameter sweeps;
they are retained here for historical reference and are not part of the active
production pipeline.
Each listed file was originally a repository-root script with the same
basename. Its original module docstring remains the detailed purpose record.

## `wfi_diagnostics/`

- `analyze_all_detector_metadata.py`
- `analyze_wfi_all_leds.py`
- `analyze_wfi_cycle_age_rate.py`
- `analyze_wfi_cycle_start_thickness_tiebreaker.py`
- `analyze_wfi_detector_acceleration.py`
- `analyze_wfi_flat_timeseries.py`
- `analyze_wfi_led1_uncal.py`
- `analyze_wfi_rate_jump_progressive.py`
- `analyze_wfi_relative_response.py`
- `analyze_wfi_relative_response_rate_bump.py`
- `analyze_wfi_temperature_rate_step.py`
- `analyze_wfi_temperature_rate_step_systematic.py`
- `analyze_wfi_thickness_overlap_rate.py`
- `analyze_wfi_tile136_ar_identifiability.py`
- `analyze_wfi_tile_acceleration.py`
- `analyze_wfi_wavelength_response_cache.py`
- `plot_wfi01_fixed_d0_comparison.py`
- `plot_wfi01_rate_block_overlay.py`

## `sweeps/`

- `sweep_wfi01_smoothness.py`
- `sweep_wfi_tile136_maxent_sensitivity.py`

## `wfi01_experiments/`

- `experiment_amber_wfi01_12_epochs.py`
- `experiment_joint_wfi01_per_tile_ice.py`
- `fit_wfi01_ar_prior_trajectory_test.py`
- `fit_wfi01_effective_boundary_comparison.py`
- `fit_wfi01_effective_boundary_numpyro.py`
- `fit_wfi01_initial_thickness_soc.py`
- `fit_wfi01_joint_optical_soc.py`
- `fit_wfi01_single_pixel_tmm.py`
- `fit_wfi01_wfi10_fixed_trajectory_test.py`
- `plot_wfi01_single_pixel_tmm_emcee.py`
- `sample_wfi01_single_pixel_tmm_emcee.py`

The WFI01 experiment cluster was moved together to retain its local helper
relationships. Before archiving, Python import-graph and source/documentation
reference scans found no production-pipeline or active-test dependencies on
these files. Shared joint-fit, modulation, bundle, and production plotting
modules are retained in the `wfi_pipeline/` implementation package.

## `legacy_root_scripts/`

This directory contains the remaining former root-level research, comparison,
one-off export, and standalone plot scripts. Their original basenames and
module docstrings are preserved. Tests tied only to those archived scripts are
stored with an `archived_test_` prefix so normal test discovery ignores them.
The active repository root now has only the two model-pipeline frontends; the
active implementation is organized under `wfi_pipeline/` and regression tests
under `tests/`.

## `cluster/`

The old Slurm fit and submit helpers are retained here as historical launch
recipes; the supported local pipeline commands do not use them.
