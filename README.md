# Roman WFI Ice Analysis

This repository contains the Roman WFI ice-monitoring analysis scripts. The
primary workflow converts calibrated MAST products into compact tile
sufficient statistics, then fits an air/ice/AR/HgCdTe transfer-matrix model
to the statistics with NumPyro.

The repository root exposes two local model-pipeline commands:
`run_wfi_relative_response_pipeline.py` and `run_wfi_full_tmm_pipeline.py`.
The fast `update_ice_history.py` utility updates full per-tile histories from a
base H5 and corrected statistics without rerunning TMM or the CAR190/Amber joint
optimizer. Shared implementation and stage runners live in `wfi_pipeline/`;
retained standalone research scripts live in `archive/exploratory/`.

The full-TMM production model is intentionally fixed. The pipeline invokes its
internal `wfi_pipeline/run_wfi_production.py` stage as fresh-EMCEE-initialized
independent per-tile MAP fits:

- corrected `*_tile_log_statistics_<tile-size>_count_corrected.csv` inputs,
  with 256 pixels as the production default tile size and no correction table
  reapplied;
- absolute-substrate-power response with semi-infinite HgCdTe;
- independent positive HgCdTe n and k log scales with prior sigma `0.25`;
- energy-weighted core SED integration using the pinned local reference files;
- exactly three independent lossless AR layers at fixed normal incidence;
- fixed lossless ice index `n=1.25`, with decon2 anchored at relative day
  `12.907397928240743`, the first unified Program 1063 epoch;
- independent per-tile lognormal initial and reset thickness priors, median `10` nm and log sigma `1.5`;
- fully non-parametric, unwhitened per-tile trajectories with a `12` nm/day lognormal rate prior;
- one fitted post-reset log offset per tile, with prior sigma `0.1`;
- deterministic least-squares MAP optimization in prior-width coordinates for up to `10000` evaluations;
- practical MAP convergence at `1e-9` function-reduction and step tolerances, with strict scaled-gradient convergence reported separately.

The initializer is fixed at 52 walkers, 1,000 steps, seed `20261004`, no
complex-q parameters, and a constant-in-time accumulation rate. Its best state
is expanded across the non-parametric rate nodes for the MAP stage. Production
has no model-selection CLI in the default full-TMM path: conductive or finite
HgCdTe, fitted optical-index branches, QE/ice modulation, ACM refits, shared
or hierarchical tile trajectories, neighboring-tile smoothing, NUTS, and
alternate MAP optimizers are not production features. An explicit
`--response-cache-root` dispatches to the definitive six-LED-integrated cached
production branch documented in `docs/WFI_RELATIVE_RESPONSE_PRODUCTION.md`.
That branch keeps the selected pinned cache's fixed optical state and interpolated response
curves, but fits the current statistics with the current production ice/rate
objective. It uses only a robust detector-level median of the source rates as a
constant warm start; it never imports the source fit's time-dependent rate
trajectory or entropy regularization, and the fitted rate prior remains centered
at `12 nm/day`.

Future cached relative-response fits should use the detector-specific six-LED
cache generated from the complete 20261109 full-TMM run:

```text
20261109/corrected_statistics_4h_thinned/production/relative_response_cache/
  tile_size_256/sensitivity_hgcdte_n_sigma_0p25/wfiNN_400nm/
```

Pass that cache with `--response-cache-root` and
`--allow-external-cache-source`. The portable all-wavelength bundle is a
separate downstream product and is not an input to the cached production fit.

The standard full-TMM production model leaves per-LED log offsets disabled
because the LED baselines are normalized. The current validated cached
relative-response run on 20261111 explicitly fits them with
`--fit-led-log-offsets` and `--led-log-offset-prior-sigma-log 0.01`; its
archives remain in the separate `relative_response/` subtree and must not
replace the standard full-TMM outputs.

The production forward model, parameterisation, priors, and regularisation
are documented in `docs/WFI_TMM_PRODUCTION_MODEL.md`.

The exploratory
`archive/exploratory/legacy_root_scripts/fit_wfi_tmm_numpyro.py` workflow
retains additional optical sensitivity branches. Those branches are not
accepted by the production pipeline or output contract.

The reset-validation workflow below is deliberately separate and retains its
own raw-input, fixed-correction, and complex-index configuration.

## Environment

The pinned base dependencies are in `requirements-wfi01-ice.txt`. The
NumPyro fit additionally requires working `jax`, `numpyro`, and `corner`
installations. The current development environment is named `bulge_env`.
`roman-mast-tools` 0.9.1 is not published on PyPI, so it is intentionally not
in the requirements file; install it separately from the supplied local source
archive after extracting it.

The default production TMM computes its own transfer-matrix response on the
exact wavelength nodes in the pinned SED tables, covering approximately
480--2700 nm. Older SOC/effective-boundary tools still use their separate model
tables. The opt-in cached relative-response branch uses a provenance-checked
six-LED cache and does not integrate the generic wavelength bundle during each
fit evaluation. The current cache is the detector-specific `20261109`
six-LED cache described in `docs/WFI_RELATIVE_RESPONSE_PRODUCTION.md`; ordinary
reruns should reuse it rather than rebuild it. The `20261020` cache is retained
only for historical reproduction. A new cache is expected only
after full reprocessing, a reduction improvement, or a deliberate model
change. Data without temperature-dependent gain correction is a future
experiment and is not supported by the current cache path.

```bash
python -m pip install -r requirements-wfi01-ice.txt
python -m pip install /path/to/roman-mast-tools-master
python -c 'import roman_telem; print(roman_telem.__file__)'
```

The final command must succeed before running the RRN pipeline. If a previous
build of the local checkout installed metadata without the module, remove its
`build/`, `dist/`, and `roman_mast_tools.egg-info/` directories and reinstall
from the cleaned checkout.

## Inputs

`all_wfi_rcs_flat_index.csv` is the tracked 4,626-product inventory used by
the streaming workflow. Each of WFI01 through WFI18 contains 257 exposures,
covering 42 complete six-LED observation epochs plus the first five products
of the next epoch.

`wfi01_tile_log_statistics_512.csv` is the tracked WFI01 fixture used by the
tests and for single-SCA smoke checks. Raw MAST products, API tokens, partial
downloads, and generated analysis products are intentionally not versioned.
The streaming tile rows also retain compact count-space sufficient statistics
for the valid reference/current pixel pairs, so tile-level count means,
variances, and covariance can be recovered without retaining image files.

## Workflow

### 1. Stream tile statistics

The downloader keeps one LED baseline and one current exposure in memory. It
writes a checkpoint per exposure and atomically publishes each completed SCA
CSV. It does not write raw ASDF products to disk.

```bash
python build_wfi_tile_statistics_stream.py \
  --all-detectors \
  --index-path all_wfi_rcs_flat_index.csv \
  --output-dir wfi_tile_log_statistics_512
```

The default behavior resumes existing `.parts/group_*.csv` checkpoints and
skips completed detector CSVs. Use `--no-resume` only when intentionally
recomputing existing statistics. The all-SCA fit requires 18 completed final
CSV files; partial checkpoint directories do not count.

When the MAST inventory is refreshed, use `--incremental` instead of
`--no-resume`. Incremental mode matches existing statistics by MAST file-set
name, downloads only newly discovered exposures (and the baseline images needed
to compare them), and preserves the existing sufficient statistics. It also
handles new exposures inserted into the middle of a LED sequence.

```bash
python query_rcs_flats.py \
  --all-detectors \
  --output all_wfi_rcs_flat_index.csv
python build_wfi_tile_statistics_stream.py \
  --all-detectors \
  --index-path all_wfi_rcs_flat_index.csv \
  --output-dir wfi_tile_log_statistics_512 \
  --incremental
```

These local archive commands retain the older Program 1039-only fixture
workflow. They are not the production path for the unified Program 1039 plus
Program 1063 reduction; use the Nexus pipeline below for that reduction.

The MAST token is read from `mast_api_token.txt` in the repository root.

### 1a. Nexus S3 reduction and correction pipeline

This reducer runs on Nexus only. See
[`docs/WFI_NEXUS_REDUCTION_HANDOFF.md`](docs/WFI_NEXUS_REDUCTION_HANDOFF.md)
for its data handoff to the local model pipeline.

On the Nexus, use the one-step pipeline instead of copying products from a
user-owned CAR190 directory. It lists calibrated ASDF products under the
CAR190 S3 prefix, refreshes one unified all-detector MAST index for Programs
1039 and 1063, explicitly fetches PD2 photodiode telemetry for each program,
uses the first non-warm-up dark-subtracted signal per LED as the shared
reference, fetches the sporadic `WFI_MCE_ADC_WTSM00` MPA/CPR-strap
temperature telemetry, uses the MAST supplement by default, and writes each
run below a UTC `YYYYMMDD` directory.
Image products are streamed into memory only; the pipeline does not create an
S3 image cache.

The Nexus deployment needs `run_wfi_reduction_pipeline.py`,
`fit_wfi_side_swap.py`, `wfi_side_swap.py`, and the repository-root
`mast_api_token.txt`; install the base packages in
`requirements-wfi01-ice.txt` and the local `roman-mast-tools` checkout
separately. The standalone script does not import the other reduction modules
or require local ASDF products, index files, telemetry CSVs, or correction
CSVs.

```bash
./run_wfi_reduction_pipeline.py --detector WFI01
```

Omit `--detector WFI01` to reduce every WFI detector, or replace it with one
detector identifier such as `WFI02`. The source, default output root, MAST token
path, MAST supplement, PD2 telemetry channel, and unified program set are fixed
production defaults. The tile edge length defaults to 512 pixels and can be
overridden with `--tile-size`; for example, `--tile-size 256` emits
256x256-pixel tiles. Only the newest available `raw_statistics/`
file is reused by file-set name; existing scaled `statistics/` files are never
used as inputs. Newly discovered products are streamed and reduced, and changed
or removed source products trigger a safe full rebuild. At most the
baseline/current image pair for the active LED is held in memory. The default
statistics output is the `512x512` tile grid, which is 64 tiles per detector.
A `256x256` override produces 256 tiles per 4096x4096 detector and is for
separate size-specific products. The production fit and plotting chain accepts
the matching layout when `--tile-size 256` is passed to its entrypoints.
Pass `--workers N` to reduce multiple detectors concurrently; each worker has
its own MAST session and the default remains one worker for conservative
single-detector operation. For example, use `--workers 18` when reducing the
full WFI set on a 32-core host.

The count-space temperature correction uses five-minute-binned
`WFI_MCE_ADC_WTSM00` telemetry evaluated through one frozen pure quadratic per
tile in log-correction space. The default coefficient table contains both
256-pixel and production 512-pixel tile resolutions. The default coefficient table is
`temperature_quadratic_coefficients.csv`; override it with
`--temperature-quadratic-coefficients PATH`. If
`--mpa-temperature-telemetry PATH` is omitted, the pipeline queries the MAST
EDP sporadic engineering stream and writes the long-form telemetry to the
dated `telemetry/` directory. A supplied CSV must contain `ObsTime`, `EUValue`,
and `mnemonic` columns and must bracket every indexed exposure. The archived
coefficient table contains one detector/tile-resolution parabola and is not
refit during a reduction. Rebuild it from the archived 256-pixel statistics
with `python archive/exploratory/legacy_root_scripts/build_wfi_temperature_quadratic_coefficients.py` when the archive
or operational-temperature timestamp changes.
When the next decon timestamp is known, pass `--decon-time ISO_TIMESTAMP`.
The pipeline then fits a free local count-space step over a two-day window,
using a time trend and WTSM00 when available; it does not assume a fixed
post-decon baseline. If there are not observations on both sides, no decon
correction is emitted.

By default, the pipeline searches the default root for the most recent dated
`raw_statistics/` output and reuses it by file-set name. Existing scaled
`statistics/` files are ignored and regenerated with current telemetry and
corrections. Changes to photodiode telemetry therefore do not force raw image
reprocessing. Use `--run-root PATH` only when the existing reduction tree is
outside the repository checkout.

When the image/source timeline, WTSM00 telemetry, temperature coefficients, and
correction settings are unchanged, pass
`--reuse-corrections-from /path/to/previous/YYYYMMDD` to reuse that run's
side-swap and count-space correction tables. The pipeline still refreshes the
photodiode telemetry, rewrites the side-swap-scaled statistics, and applies the
frozen count corrections to the new statistics. Reuse is refused if the
non-photodiode source/timeline fingerprint, WTSM00 values, correction settings,
or correction row keys differ. A run must have been produced by the cache-aware
pipeline once before it can be reused.

The archived temperature correction was calibrated from the nonzero correction
points and is applied to every production count-statistics row with valid
WTSM00 telemetry, without refitting. The fitted curve is normalized so its
correction is exactly zero at the operational temperature; its original
intercept is retained in the correction summary for provenance. The same
per-tile quadratic is used before and after a future decon, while the decon
itself remains a separate free count-space step. The legacy time sinusoid is
retained only for embedded callers that explicitly omit temperature telemetry.
The previous
`15.064405023148149`-day anchor is the
operational-temperature transition used to partition the ACM fit. The
ACM repeat boundaries remain relevant to the ACM fit and legacy fallback. The
repeat-669 additional
ACM step and the day-22 step (effective repeat `682` in the 20261009 WFI02
timeline) are fitted independently for each LED; the other ACM steps remain
shared grey terms. The legacy post-operational fit ends before day 22;
the per-LED day-22 step is estimated locally from neighboring reference
observations rather than absorbed into that fallback. A later day-26 ACM
adjustment is mapped to effective repeat `691` and fitted as one shared-grey
local step across the LEDs. The physical ice reset is
the first Program 1063 observation and is recorded per detector in
`pipeline_manifest.json`. The existing ACM and side-swap corrections remain
separate fields in the same count-space correction table.

Each dated directory contains:

- `statistics/`: per-detector side-swap-corrected photodiode-scaled tile
  sufficient statistics, including count moments;
- `raw_statistics/`: the same tile statistics computed from truly raw,
  uncorrected detector counts;
- `corrections/`: per-detector side-swap calibration files plus fitted
  count-space gain/ACM correction tables and summaries;
- `corrected_statistics/`: the photodiode-scaled tile statistics with the
  fitted count correction applied; the filename records the tile size;
- `diagnostics/`: scaled-versus-corrected current-count plots for each detector,
  including LED3 pre- and post-operational temperature-fit diagnostics. To plot
  the per-exposure photodiode scale recorded in an inventory, run
   `python archive/exploratory/legacy_root_scripts/plot_wfi_photodiode_correction.py --inventory PATH --output PATH`;
  the script also writes a tidy CSV next to the PNG. Use `--quantity percent`
  to plot departure from unity instead of the scale factor. Pass
  `--side-swap-calibration PATH` to mark the fitted per-LED side-swap event;
- `telemetry/`: dark-subtracted per-exposure PD2 summaries fetched separately
  for Programs 1039 and 1063 by explicit `roman-telem --ice-monitor
  --ice-program PROGRAM` calls before reduction, plus the five-minute-binned
  sporadic `WFI_MCE_ADC_WTSM00` engineering series used by the count-space
  temperature fit; the first non-warm-up signal per LED across the PD2
  summaries is the shared production reference;
- `pipeline.log`: timestamped phase, detector, and product progress;
- `all_wfi_rcs_flat_index.csv`: refreshed MAST supplement index;
- `*_rrn_inventory.csv`: the source products and timestamps used;
- `pipeline_manifest.json`: source, command, detector, correction, and output
  provenance.

The side-swap-corrected photodiode-scaled statistics plus the count correction
table remain the canonical count-correction inputs. `raw_statistics/` remains
truly raw for auditing and independent recalculation. Do not pass `*_count_corrected.csv`
together with that same correction table, or the count correction will be
applied twice. The pipeline's `*_count_corrected.csv` files are provided as
explicitly corrected products for inspection and downstream tools that do not
apply the table themselves.

The pipeline stops if the root token, MAST index query, telemetry query, or
MAST supplement cannot be refreshed; it does not silently fall back to a
partial local-only reduction.

### 1b. Optional Program 1063 cadence thinning

The canonical `corrected_statistics/` files retain every corrected epoch. When
the high-cadence Program 1063 sequence makes the production trajectory fit
unnecessarily difficult, create a derived input directory without changing
those canonical files:

```bash
.venv/bin/python wfi_pipeline/thin_wfi_statistics.py \
  --source-dir /path/to/corrected_statistics \
  --output-dir /path/to/corrected_statistics_program1063_1h \
  --spacing-hours 1
```

This copies all Program 1039 rows unchanged and retains complete six-LED
Program 1063 cycles separated by at least one hour. The output includes
`program1063_cadence_thinning_manifest.json`, which records the selected
repeats and row counts. The thinning operates on already corrected sufficient
statistics; it does not average rows or apply another correction.

Use the derived directory as input to the relative-response or full-TMM
pipeline frontend only after
checking the manifest. The source corrected statistics remain the canonical
product for auditing and for any analysis that needs the full cadence.

### 1c. Local model-to-product pipeline

After transferring the corrected statistics from Nexus, run one of the
path-driven local frontends. Relative response is the normal, lower-cost route;
it reuses the detector-specific pinned 20261109 six-LED cache. Full TMM is the
occasional, more expensive alternative and builds its six-LED and wavelength
response caches from that run's fits. Each command selects only one model route
and accepts the statistics/run-root path plus a dedicated output path:

```bash
python run_wfi_relative_response_pipeline.py \
  --input-path /path/to/20261113/corrected_statistics_4h_thinned \
  --output-path /path/to/20261113/production/pipeline_runs/relative_response

python run_wfi_full_tmm_pipeline.py \
  --input-path /path/to/20261113/corrected_statistics_4h_thinned \
  --output-path /path/to/20261113/production/pipeline_runs/full_tmm
```

`--input-path` may instead be the dated Nexus run root containing
`corrected_statistics/` or `corrected_statistics_4h_thinned/`. Unthinned
corrected statistics are thinned locally to four-hour cadence and 256-pixel
tiles; the existing count corrections are not applied again. The output path
must be a dedicated directory under the same dated data folder, outside the
statistics directory.

For the cached relative-response route, transfer both the pinned 20261109
six-LED response caches and their provenance source files: the corresponding
`production/wfi##/fits/` and `production/wfi##/tile_statistics/` trees from that
run. The cache uses those fit archives for its optical initialization and rate
warm start; the spectral-response grid is also built from those original TMM
optical fits, not from the ice-only relative-response fit outputs. Cache
provenance paths are resolved against the current repository root and their
hashes are checked, so a relocated checkout works when these dated source files
are present. Copying the response-cache directory alone is not sufficient.

Each run owns its `pipeline_manifest.json`, central `logs/`, route-specific
fits and caches, the stable `wfi_all_2nm_rank16.npz` and
`wfi_all_2nm_rank16_modulated.npz` portable bundles, clearly separated H5 build
intermediates, one final H5 plus validation report, and TMM/CAR201 plots. The
manifest fingerprints its input CSVs and records
resolved model defaults, detector products, stage logs, and hashes for the final
bundles/H5. A matching incomplete run resumes from completed stages; a changed
input, route, settings profile, or pipeline version is rejected for that output
path. See `docs/WFI_FULL_PIPELINE_REFACTOR_PLAN.md` for the output layout and
regression contract.

The deliverable H5 is `production/wavelength_response_bundle/final/joint_modulated_v2.h5`.
The reflectance-mode, history-reference, and prototype H5s are build-stage
intermediates under `production/wavelength_response_bundle/intermediates/h5/`.

### Fast H5-based full ice-history update

Use this when new corrected statistics arrive and the H5's fixed optical state
should be retained. The updater derives the six-LED response from the supplied
H5, then fits independent 48-hour log-rate splines with knots at the reset
boundaries, plus reset thicknesses, grey offsets, and LED offsets. Tile fits
are evaluated in independent accelerator batches; there is no cross-tile
smoothing or shared trajectory. The response table is cached by the H5 optical
fingerprint. No source per-tile fits, fixed-response cache, TMM fit, or
CAR190/Amber joint fit is used by the ice estimator.

```bash
python update_ice_history.py \
  --input-path /path/to/Nexus/20261113 \
  --base-h5 /path/to/Nexus/20261111/production/wavelength_response_bundle/final/joint_modulated_v2.h5 \
  --output-path /path/to/Nexus/20261113/production/pipeline_runs/relative_response_update
```

`--input-path` resolves the 18 corrected-statistics files. `--base-h5` is an
already-existing H5 from the previous validated run; it is independent of the
new statistics and does not need to be produced in the new dated directory.
The fit uses that H5 as its only optical model input; no per-tile fit archive
or fixed-response cache is read. Standard wavelength bundles beside the base
H5 are preserved as spectral-output scaffolds. LED offsets are fitted from the
corrected statistics. Batch size, 48-hour spline spacing, and iteration limits
use internal defaults. It writes the usual base/modulated bundle and
`final/joint_modulated_v2.h5` filenames.
The complete independent tile histories are retained at every data epoch in
`production/wavelength_response_bundle/intermediates/h5/joint_ice_history_reference.h5`;
the Eric-reader H5 keeps the rank-one deposition projection for compatibility.
The updater also refreshes all per-detector diagnostics and the complete 18-SCA
focal-plane and ice-trend plot sets under
`production/plots/relative_response/{focal_plane,trajectories,trajectories_last_7d}/`.

The runtime data inputs that used to live beside the repository are now copied
under `car201_data/` and `external_reference_data/`. Their `INPUTS.json` and
`PORTABLE_INPUTS.json` files record sizes and SHA256 hashes. Copy both data
folders with the repository when moving the pipeline to another machine; the
active code resolves them from the repository root and no longer uses sibling
`CAR201/` or workspace paths.

Stage modules in `wfi_pipeline/` are implementation details. Standalone
analysis and experiment scripts have been moved to `archive/exploratory/`.

### 2. Internal production TMM stage

The supported orchestration entrypoints are the two pipeline commands in
section 1c. Their full-TMM stage is implemented by
`wfi_pipeline/run_wfi_production.py`. Direct invocation is intended only for
focused development checks.
Point it at a directory containing corrected per-detector tile-statistics files:

```bash
.venv/bin/python -u \
  wfi_pipeline/run_wfi_production.py \
  --statistics-dir /path/to/corrected_statistics
```

Each input directory must contain files named
`{detector_lower}_tile_log_statistics_<tile-size>_count_corrected.csv`; 256 is
the production default and can be overridden with `--tile-size 512`. All WFI01-WFI18
detectors are processed by default; repeat `--detector WFI##` to select a
subset. Use `--tile-id 1` for a focused tile regression, and `--force` to
refit a tile whose matching archive already exists. The runner validates the
contiguous tile IDs in the input rather than assuming 64 tiles.

Each detector starts with a fresh 52-walker, 1,000-step constant-rate EMCEE
basin search (seed `20261004`, no complex-q parameters). Its best state is
expanded across the non-parametric rate nodes and initializes one independent
fit per input tile using deterministic least-squares MAP optimization for up to
10,000 evaluations. The fits use absolute semi-infinite HgCdTe power with free
positive n/k nuisance scales, independent tile AR parameters, lognormal initial
thickness priors (median 10 nm, log sigma 1.5), the 12 nm/day trajectory prior,
tile-rate and trajectory variation, and no neighboring-tile rate smoothing.
Corrections are expected to be already included in the input statistics; no
fixed-correction CSV is passed to the model. Existing tile archives are reused
only when their sufficient-statistics fingerprint matches the current input.
Independent tile MAP fits run in four spawned worker processes by default; pass
`--workers 1` for serial execution or another positive value to tune bounded
parallelism. The per-tile constant-rate warmup and final MAP remain sequential
within each worker, and detector-level aggregation waits for all workers.

All production outputs are written below the input directory, not the
repository root:

```text
<statistics-dir>/production/<detector>/       # default 256-pixel layout
  emcee/                    # one constant-rate initializer
  tile_statistics/          # one split CSV per tile
  fits/tile_000/            # one deterministic MAP archive per tile
  tile_rate_posterior.csv   # detector-level aggregation
```

For a non-default tile size, outputs are isolated under
`<statistics-dir>/production/tile_size_<tile-size>/<detector>/` so they cannot
mix with the default 256-pixel archives. Pass the same `--tile-size` to the
production plotters and bundle command.

`wfi_pipeline/fit_wfi_tmm_production.py` contains only the fixed production TMM model and
MAP writer. The independent runner is the supported entrypoint for the current
cold-EMCEE plus per-tile least-squares production configuration.

For the cached relative-response production branch, use the pinned-cache
workflow in `docs/WFI_RELATIVE_RESPONSE_PRODUCTION.md`. It writes a separate
`relative_response/` output subtree and never replaces the standard full-TMM
`fits/` archives.

The current validated cached run uses the 20261111 256-pixel corrected
statistics, a separate constant-rate warmup rooted at
`relative_response_constant_rate_warmup_start40/`, an explicit third-decon day
of `38.79784510416666`, a third-decon warmup-centered thickness prior with log
sigma `0.1`, independent one-tile fits (`--batch-size 1`), and zero trajectory
entropy weight. The final fit still performs its own scalar-rate numerical
warmup; the separate warmup supplies the per-tile third-decon prior centers.
Use the detector-specific six-LED cache from the complete 20261109 run with
`--response-cache-root` and `--allow-external-cache-source`. Do not pass
`--joint-batch` or a fixed-correction table.

Export the fitted ice state as a spreadsheet-friendly CSV with one row per
tile and shared repeat-time model node:

```bash
.venv/bin/python -u archive/exploratory/legacy_root_scripts/export_wfi_production_ice_csv.py \
  --statistics-dir 20261111/corrected_statistics_4h_thinned \
  --fit-root 20261111/corrected_statistics_4h_thinned \
  --output 20261111/corrected_statistics_4h_thinned/production/relative_response/wfi_ice_accumulation_20261111.csv
```

The primary table contains one row per detector/tile/model node and includes
`ice_thickness_nm`,
`accumulation_rate_nm_day`, nominal MPA focal-plane `mpa_x_mm` and
`mpa_y_mm` coordinates, detector/tile geometry, node metadata, reset
boundaries, fit mode, and archive provenance. The 20261111 export contains
124 shared model nodes per tile; LED exposures are not separate model rows.
Add `--current-output <path>` when a separate latest-node snapshot is useful.
When `--output` is omitted, the exporter derives the
`YYYYMMDD` filename label from the dated statistics directory; use
`--date-label` for an undated input path.

Generate matching detector and all-detector plots from the same statistics
directory with the default 256-pixel layout. Add `--tile-size 512` to each
command when plotting the alternate layout:

```bash
.venv/bin/python -u wfi_pipeline/plot_wfi_independent_production.py --root /path/to/corrected_statistics
.venv/bin/python -u archive/exploratory/legacy_root_scripts/plot_wfi_all_detector_trajectories.py \
  --root /path/to/corrected_statistics \
  --reset-day 12.907397928240743
.venv/bin/python -u archive/exploratory/legacy_root_scripts/plot_wfi_fitted_incidence_angle_mosaic.py \
  --root /path/to/corrected_statistics
.venv/bin/python -u archive/exploratory/legacy_root_scripts/plot_wfi_post_reset_initial_thickness_mosaic.py \
  --root /path/to/fit_archive_root
.venv/bin/python -u archive/exploratory/legacy_root_scripts/plot_wfi_full_focal_plane_rates.py \
  --root /path/to/corrected_statistics
```

The incidence-angle command writes
`production/wfi_fitted_incidence_angle_mosaic.png` below the statistics directory.
The post-reset initial-thickness command writes
`production/wfi_post_reset_initial_thickness_mosaic.png` and uses the complete
current production fit layout before falling back to the complete legacy layout.
The current authoritative Decon 3 boundary is relative day
`38.79784510416666`, established in the 20261023 production fit. Pass
`--reset-stage 3 --reset-day 38.79784510416666` to plot the third-decon initial
thickness, and pass `--third-decon-day 38.79784510416666` to the production
runner. Stage 3 uses a separate
`wfi_post_third_decon_initial_thickness_mosaic.png` default output name.

Isolated fit trees can be plotted without copying them into the canonical
`production/` layout. Use `--fit-root` for the post-reset, incidence-angle, and
focal-plane mosaic plotters, `--detector-output-root` for the trajectory plotter,
and the detector directory itself with `--fit-root` for
`wfi_pipeline/plot_wfi_independent_production.py`. For example, an offset branch under
`production_led_offsets/` can be plotted by selecting `WFI01` and pointing
these options at that branch. The incidence-angle plot remains empty for the
normal-incidence production model because those archives intentionally contain
no fitted incidence-angle parameter.

The trajectory plotter supports `--break-day` more than once for independent
reset boundaries. Its last-seven-day view autos-scales to the selected data;
the full-series rate view retains the historical 30 nm/day upper limit. The
focal-plane plotter can additionally write an AOM jump map with
`--aom-rate-jump-output`; the two panels compare the latest post-AOM rate with
the last pre-AOM rate in absolute nm/day and percent. Corrected tile-statistics
diagnostics use scatter points rather than connected data lines; for example:

```bash
.venv/bin/python -u archive/exploratory/legacy_root_scripts/plot_wfi_corrected_tile_neighborhood.py \
  --root /path/to/corrected_statistics \
  --detector WFI06 \
  --tile-id 98 \
  --tile-size 256 \
  --output /path/to/wfi06_tile098_neighborhood_corrected_data.png
```

### Production prior bundle

Bundle the complete production outputs into one self-describing FITS product:

```bash
.venv/bin/python -u archive/exploratory/legacy_root_scripts/bundle_wfi_production_prior.py \
  --statistics-dir /path/to/corrected_statistics \
  --fit-root /path/to/fit_archive_root
```

Add `--tile-size 256` when bundling the alternate-size fit archives. The
bundle records the discovered tile count and detector grid in its metadata.

For the current cached relative-response production branch, fit archives are
under each detector's `relative_response/fits` directory and the wavelength
grid is supplied explicitly from a fixed-optics spectral cache:

```bash
.venv/bin/python -u archive/exploratory/legacy_root_scripts/bundle_wfi_production_prior.py \
  --statistics-dir 20261111/corrected_statistics_4h_thinned \
  --fit-root 20261111/corrected_statistics_4h_thinned/production \
  --response-grid-cache 20261109/corrected_statistics_4h_thinned/production/wavelength_response_cache/tile_size_256/wfi01/tile_000/spectral_response.npz \
  --output 20261111/corrected_statistics_4h_thinned/production/wfi_production_map_prior_bundle.fits
```

The resulting FITS product records the cached fit mode and external response
grid source in its `MODEL` extension.

The default output is
`production/wfi_production_map_prior_bundle.fits` below the statistics directory.
The primary header and embedded `README`, `MODEL`, and `SCHEMA` extensions define
the time, position, units, join keys, fit estimator, and provenance. The product
contains detector positions, zero-based tile pixel bounds, UTC ISO exposure-start
timestamps, UTC MJD timestamps, relative model days, fitted parameters, fitted
observation-time trajectories, non-parametric node trajectories, residuals, and
source/archive checksums. It records MAP point estimates, not posterior
uncertainties. The bundle exposes both the raw production AR thickness
coordinates and physical AR thickness in nm; physical thickness is exactly
`100 * raw_coordinate`.

The `HGCDTE_REFERENCE` extension contains the wavelength-dependent internal
reference `n` and `k` curve used by the production TMM on its response grid.
`PARAMETERS.absolute_hgcdte_n` and `absolute_hgcdte_k` contain the reconstructed
physical per-tile vectors; the corresponding log-scale columns are retained as
the fitted prior parameters. Reconstruction is
`n = reference_n * exp(n_log_scale)` and
`k = reference_k * exp(k_log_scale)`. The bundled reference is the internal
80 K, `x=0.445` sensitivity model, not a detector-specific validation table.
The production absolute-substrate-power response does not multiply the
independent measured QE values; a downstream model using Roman system
effective area must apply that system response once instead.
When a third-decon fit is supplied, the bundle additionally carries the third
boundary and per-tile post-third-decon thickness/grey-offset fields; those
fields are absent from legacy bundles.

The legacy `archive/exploratory/legacy_root_scripts/fit_wfi_tmm_numpyro.py`
implementation is retained for historical reproduction and is not a supported
pipeline entrypoint.

### WFI01 reset validation

The validated WFI01 pre/post complex-index reset fit below is a raw-input
reproduction, separate from the corrected-statistics production workflow above. It uses
the raw RRN tile statistics and the grey-corrected count-space correction
table. Do not substitute
`wfi01_tile_log_statistics_512_count_corrected.csv`: the fixed table already
contains the `count_log_correction` applied to the raw statistics.

The known-good one-tile command is in
`wfi01_absolute_substrate_power_grey_corrected_complex_ice_map5000_one_tile/numpyro_summary.md`.
For all 64 WFI01 tiles, use the same command with only `--max-tiles 1` removed
and the output directory changed:

```bash
.venv/bin/python3 archive/exploratory/legacy_root_scripts/fit_wfi_tmm_numpyro.py \
  --single-sca \
  --tile-statistics rrn_reduction_20260930/wfi01_tile_log_statistics_512.csv \
  --fixed-corrections rrn_reduction_20260930/wfi01_count_space_fixed_corrections_three_events_grey_corrected.csv \
  --absolute-substrate-power \
  --sed-weighting energy \
  --initial-thickness-prior-mean-nm 0 \
  --initial-thickness-prior-sigma-nm 50 \
  --initial-thickness-init-nm 1e-6 \
  --trajectory-rate-prior-mean-nm-day 5 \
  --trajectory-rate-prior-sigma-log 1.5 \
  --ar-mean-prior-sigma 10 \
  --ar-sca-sigma-prior 1 \
  --ar-tile-sigma-prior 1 \
  --map-trajectory-rate-nm-day 10.2 \
  --map-parameter-scaling prior \
  --map-optimizer lbfgs \
  --tile-rate-variation \
  --tile-rate-trajectory-variation \
  --no-tile-rate-neighbor-smoothness \
  --fit-incidence-angle \
  --incidence-angle-prior-uniform \
  --incidence-angle-init-deg 7.5 \
  --ice-reset-day 15.064405023148149 \
  --fit-pre-post-ice-index \
  --ice-extinction-prior-sigma 0.005 \
  --ice-extinction-upper 0.05 \
  --additional-acm-step-repeat 39 \
  --pre-reset-acm-step-repeat 35 \
  --prediction-row-chunk-size 64 \
  --map-steps 5000 \
  --map-only \
  --no-progress-bar \
  --output-dir wfi01_absolute_substrate_power_grey_corrected_complex_ice_map5000_all_tiles
```

The legacy plotting companion is not included in this checkout; keep the
historical fit output and summary for reproduction provenance.

All reset-validation outputs and logs stay inside the repository workspace.

### Legacy NumPyro comparison workflow

The following options apply only to
`archive/exploratory/legacy_root_scripts/fit_wfi_tmm_numpyro.py` comparison runs.
They are not supported by the fixed production entrypoint.

Use `--ar-layers 1` or `--ar-layers 2` for comparison models. If the default
output path is used, they are written to paths ending in `_ar1layer` and
`_ar2layer` rather than overwriting the three-layer fit. Layer order is from
air toward HgCdTe.

For the reduced all-tile model, add `--shared-ar-optical-properties`. This
shares each layer's refractive index across every tile while retaining
hierarchical tile-to-tile thickness variation. The dielectric model fixes
`k=0` by default; `--fit-ar-absorption` is an explicit sensitivity option.

The general CLI settings include a three-layer dielectric AR stack, NUTS
sampling, all available tiles, exact SED wavelength nodes and stride 1, and
log-noise sigma 0.005. The likelihood pixel weight defaults to the inverse median
valid-pixel count, converting each tile/group pixel SSE to approximately one
effective datum before the selected-tile average. The aggregate SSE is then
normalized by the number of selected tiles so adding tiles does not make NUTS
geometrically intractable. Use `--likelihood-weight` only for an explicit
sensitivity run.

Use `--fixed-corrections PATH` to apply fixed gain and first/late ACM
corrections to raw comparison statistics. The fixed production workflow does
not pass this option because its corrected input already includes those terms.

For a prior-sensitivity run, `--ar-mean-prior-sigma 5` makes the global AR raw
parameter prior much broader than the default. This is useful for the focused
one-tile diagnostic, but should not be treated as a final physical prior.
Trajectory-rate sensitivity can be run without changing the likelihood, for
example `--trajectory-rate-prior-mean-nm-day 10
--trajectory-rate-prior-sigma-log 1`. The rate prior is placed on the
unmodulated instantaneous rate at the first observed time. The model integrates
that positive rate to obtain ice thickness; with `--np-modulation`, epoch-node
multipliers modulate the instantaneous rate before integration.
Use `--tile-rate-variation` to give each selected SCA/tile its own log-rate
offset around the shared trajectory. Add `--tile-rate-trajectory-variation`
to give every tile independent log-rate deviations at each observation node;
comparison runs may use both options. Neighboring selected tiles receive a
Gaussian smoothness prior based on their row/column coordinates unless
`--no-tile-rate-neighbor-smoothness` is supplied. The tile rate prior, neighbor
prior, and time-curvature scales are controlled by
`--tile-rate-prior-sigma-log`, `--tile-rate-neighbor-sigma-log`, and
`--tile-rate-trajectory-smoothness-weight`.
The default prior/start is an equal-index-ratio quarter-wave dielectric stack
at 1050 nm with `k=1e-4`. Use `--ar-prior-from-map-start` to recenter that prior
when selecting a different physical start.

The exact-node grid uses about 3,294 wavelength nodes for the current public
SED tables. Use `--wavelength-step-nm 0.5` to test a uniform sub-nanometer
grid; keep `--wavelength-stride 1` for that comparison. In
`--absolute-substrate-power` mode, QE is disabled and the response integrator
automatically removes exact-zero-weight nodes after integration. This is an
algebraic sparsity optimization, not a wavelength approximation; the retained
grid is recorded in `posterior_samples.npz` and `numpyro_summary.md`.

CPU chains run sequentially by default. `--vectorized-chains` runs both chains
as one batched JAX computation without creating virtual CPU devices and is the
preferred faster option for the full fit. Use `--parallel-chains` only for
smaller fits or after confirming that host-device contention is acceptable.

Use `--map-steps N` to run `N` full-model AutoDelta optimization steps before
NUTS. MAP is only an initializer; posterior sampling and all model parameters
remain unchanged. `--map-learning-rate` controls its Adam step size; the
lowest checkpointed MAP objective is retained if a long Adam run later drifts.
Use `--map-only` with `--map-steps` to archive and plot the deterministic MAP
solution without starting NUTS; this is useful for fast basin comparisons.
The deterministic MAP optimizer defaults to bounded L-BFGS-B; use
`--map-optimizer adam` only for the legacy AutoDelta initializer path, and use
`--map-tolerance` to set the selected MAP optimizer's projected-gradient
stopping tolerance. `--map-optimizer trust-exact` is an optional exact-Hessian
trust-region polisher for small, warm-started fits; it holds coordinates already
at active bounds fixed, requires `--map-parameter-scaling none`, and is not
intended for full production tile fits.
For a difficult one-tile fit, first archive a good `lbfgs` MAP state, then pass
its output directory to `--map-init-from` with `--map-optimizer trust-exact`.
The reparameterized AR and whitened-trajectory sites are converted from their
saved physical deterministic sites when loading such a warm start.
Use `--map-trajectory-rate-nm-day` for a deliberately dispersed MAP start when
checking for separate trajectory basins; it does not alter the posterior prior.
For a staged start, `--map-init-max-tiles 1` runs those MAP steps first on one
tile per SCA, transfers the global and SCA-level sites, and then refines the
full selected-tile model. Full-tile offsets start at zero rather than copying
the single representative tile.
For a physically motivated optical start, `--map-design-wavelength-nm 1050`
uses quarter-wave optical thickness in each layer and equal log-index steps
between air and HgCdTe only during MAP initialization; it does not constrain
the sampled posterior. For the analytic three-layer equal-ratio start, add
`--map-middle-layer-eighth-wave`; this uses lambda/4, lambda/8, lambda/4 optical
thicknesses from air toward HgCdTe only for the MAP start.

For a legacy single-SCA fit:

```bash
python archive/exploratory/legacy_root_scripts/fit_wfi_tmm_numpyro.py \
  --single-sca \
  --tile-statistics wfi01_tile_log_statistics_512.csv \
  --max-tiles 1 \
  --map-steps 1000 \
  --dense-mass \
  --output-dir wfi01_tmm_numpyro_tile_pilot
```

### 2a. Run the two- and three-layer fits on Slurm

The repository includes a one-node Slurm job that runs the two-layer and
three-layer all-SCA fits concurrently, plus an `rsync`/`ssh` helper that pushes
the source and statistics before submitting it. See
[`docs/cluster_slurm.md`](docs/cluster_slurm.md):

```bash
export CLUSTER='user@login.example.edu'
export REMOTE_ROOT='/path/on/cluster/wfi-ice-analysis'
bash cluster/push_and_submit.sh
```

The optional `--np-modulation` flag adds an entropy- and smoothness-regularized
epoch-node modulation of the instantaneous rate. It is a diagnostic option,
not the default model.

### 3. Plot posterior products

Legacy NumPyro plotting scripts are retained only when their source remains in
`archive/exploratory/`; the supported model pipeline writes its diagnostics
under the selected run output path.

## Outputs

Each production tile directory contains:

- `posterior_samples.npz`: deterministic MAP sites, fixed production settings,
  and sufficient-statistics fingerprints;
- `tile_group_predictions.csv`: observed and modeled group log changes;
- `tile_ar_posterior.csv`: AR parameter summaries per tile;
- `tile_hgcdte_posterior.csv`: independent HgCdTe n/k log-scale summaries;
- `tile_rate_posterior.csv`: initial, final, and mean trajectory-rate summaries;
- `map_summary.md`: optimizer diagnostics and the fixed production settings.

The detector directory also contains the one-tile EMCEE archive and summary
used to initialize its MAP fits. The older NumPyro plotter documents and plots
the separate legacy output format.

## Code Map

- `run_wfi_relative_response_pipeline.py`: default, pinned-cache route;
- `run_wfi_full_tmm_pipeline.py`: occasional full-TMM route;
- `wfi_pipeline/`: shared model, fit, bundle, H5, and plotting implementation;
- `tests/`: active regression tests;
- `archive/exploratory/`: retired standalone analyses, comparisons, and legacy
  commands, indexed in `archive/exploratory/README.md`.

## Portable Wavelength Response

`docs/WFI_WAVELENGTH_RESPONSE_BUNDLE.md` documents the transportable compressed
wavelength-space response product, including its nominal MPA focal-plane tile
coordinates and standalone reader examples. The generated all-detector bundle
is written under the selected run's `production/wavelength_response_bundle/`
directory.
The bundle also carries
the fitted MAP ice thickness and rate at each shared observation day and UTC
exposure-start time for each tile.
The bundle also carries the fitted post-reset grey throughput offset and its
multiplicative factor.

The portable wavelength bundle is a separate downstream product. It is not the
production cache input; production cached fits use the provenance-controlled
six-LED relative-response caches documented above.

## Verification

Run the tests from the project root:

```bash
python -m pytest -q tests
python -m py_compile run_wfi_relative_response_pipeline.py run_wfi_full_tmm_pipeline.py
```

The tests do not require raw MAST products. The MAST streaming tests use
injected image fetchers and the tracked index/statistics fixtures.

## Caveats

- The all-SCA implementation selects `SCU 1` through `SCU 18` from the public
  QE table for `WFI01` through `WFI18`, respectively. The official SCU-to-SCA
  serial mapping identifies WFI01/SCU 1 with SCA serial 22081.
- The production response uses the pinned SED nodes and an effective 80 K,
  `x=0.445` HgCdTe boundary; those constants are not detector-specific
  measurements.
- The three-layer AR stack is an effective dielectric model, not a claim about
  the proprietary physical coating recipe.
- The exact external effective-boundary equation is not available, so the
  effective-boundary scripts remain sensitivity diagnostics.
- Short smoke chains are useful for exercising code paths but are not
  scientifically interpretable posterior fits.
- Generated outputs and raw products are ignored by Git; retain important
  results separately if they need to be archived.
