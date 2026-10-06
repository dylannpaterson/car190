# Roman WFI Ice Analysis

This repository contains the Roman WFI ice-monitoring analysis scripts. The
primary workflow converts calibrated MAST products into compact tile
sufficient statistics, then fits an air/ice/AR/HgCdTe transfer-matrix model
to the statistics with NumPyro.

The production model is intentionally fixed and is run through
`run_wfi_production.py` as fresh-EMCEE-initialized independent per-tile MAP fits:

- corrected `*_tile_log_statistics_512_count_corrected.csv` inputs, with no correction table reapplied;
- absolute-substrate-power response with semi-infinite HgCdTe;
- independent positive HgCdTe n/k log scales with prior sigma `0.25`;
- energy-weighted core SED integration using the pinned local reference files;
- exactly three independent lossless AR layers and a fitted incidence angle;
- fixed lossless ice index `n=1.25`, with reset day `15.064405023148149`;
- independent per-tile lognormal initial and reset thickness priors, median `10` nm and log sigma `1.5`;
- fully non-parametric, unwhitened per-tile trajectories with a `5` nm/day lognormal rate prior;
- one fitted post-reset log offset per tile, with prior sigma `0.1`;
- deterministic least-squares MAP optimization in prior-width coordinates for up to `10000` evaluations.

The initializer is fixed at 52 walkers, 1,000 steps, seed `20261004`, no
complex-q parameters, and a constant-in-time accumulation rate. Its best state
is expanded across the non-parametric rate nodes for the MAP stage. Production
has no model-selection CLI: conductive or
finite HgCdTe, fitted optical-index branches, QE/ice modulation, ACM refits,
shared or hierarchical tile trajectories, neighboring-tile smoothing, NUTS,
and alternate MAP optimizers are not production features.

The exploratory `fit_wfi_tmm_numpyro.py` workflow retains additional optical
sensitivity branches. Those branches are not accepted by the production
entrypoint or production output contract.

The reset-validation workflow below is deliberately separate and retains its
own raw-input, fixed-correction, and complex-index configuration.

## Environment

The pinned base dependencies are in `requirements-wfi01-ice.txt`. The
NumPyro fit additionally requires working `jax`, `numpyro`, and `corner`
installations. The current development environment is named `bulge_env`.
`roman-mast-tools` 0.9.1 is not published on PyPI, so it is intentionally not
in the requirements file; install it separately from the supplied local source
archive after extracting it.

The production TMM computes its own transfer-matrix response on the exact
wavelength nodes in the pinned SED tables, covering approximately 480--2700 nm.
Older SOC/effective-boundary tools still use their separate model tables; the
production TMM has no model-path compatibility option.

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

The MAST query defaults to RCS program `1039`; the statistics builder rejects
indices containing other programs because their partial LED sequences are not
compatible with the RCS time series.

The MAST token is read from `mast_api_token.txt` in the repository root.

### 1a. Nexus S3 reduction and correction pipeline

On the Nexus, use the one-step pipeline instead of copying products from a
user-owned CAR190 directory. It lists calibrated ASDF products under the
CAR190 S3 prefix, refreshes the all-detector MAST index, fetches the PD1
photodiode telemetry, uses the MAST supplement by default, and writes each run
below a UTC `YYYYMMDD` directory. Image products are streamed into memory only;
the pipeline does not create an S3 image cache.

The RRN deployment needs only `run_wfi_reduction_pipeline.py` and the
repository-root `mast_api_token.txt`; install the base packages in
`requirements-wfi01-ice.txt` and the local `roman-mast-tools` checkout
separately. The standalone script does not import the other reduction modules
or require local ASDF products, index files, telemetry CSVs, or correction
CSVs.

```bash
./run_wfi_reduction_pipeline.py
```

There are no setup arguments. The source, output root, detector list, MAST
token path, MAST supplement, PD1 telemetry channel, tile sizes, and correction
settings are fixed production defaults. Existing raw statistics are reused by
file-set name; newly discovered products are streamed and reduced, and changed
or removed source products trigger a safe full rebuild. At most the
baseline/current image pair for the active LED is held in memory. The only
statistics output is the `512x512` tile grid, which is 64 tiles per detector.

Each dated directory contains:

- `statistics/`: raw tile sufficient statistics, including count moments;
- `corrections/`: fitted count-space gain/ACM correction tables and summaries;
- `corrected_statistics/`: the same 512-tile statistics with the fitted count
  correction applied;
- `telemetry/photodiode_pd1_ice_monitor.csv`: dark-subtracted per-exposure PD1
  summary produced by `roman-telem --ice-monitor` before reduction;
- `pipeline.log`: timestamped phase, detector, and product progress;
- `all_wfi_rcs_flat_index.csv`: refreshed MAST supplement index;
- `*_rrn_inventory.csv`: the source products and timestamps used;
- `pipeline_manifest.json`: source, command, detector, correction, and output
  provenance.

The raw statistics plus the correction table remain the canonical TMM fit
inputs. Do not pass `*_count_corrected.csv` together with that same correction
table, or the count correction will be applied twice. The pipeline's
`*_count_corrected.csv` files are provided as explicitly corrected products for
inspection and downstream tools that do not apply the table themselves.

The pipeline stops if the root token, MAST index query, telemetry query, or
MAST supplement cannot be refreshed; it does not silently fall back to a
partial local-only reduction.

### 2. Production TMM fit

The production orchestration entrypoint is `run_wfi_production.py`.
Point it at a directory containing corrected per-detector tile-statistics files:

```bash
.venv/bin/python -u \
  run_wfi_production.py \
  --statistics-dir /path/to/corrected_statistics
```

Each input directory must contain files named
`{detector_lower}_tile_log_statistics_512_count_corrected.csv`. All WFI01-WFI18
detectors are processed by default; repeat `--detector WFI##` to select a
subset. Use `--tile-id 1` for a focused tile regression, and `--force` to
refit a tile whose matching archive already exists.

Each detector starts with a fresh 52-walker, 1,000-step constant-rate EMCEE
basin search (seed `20261004`, no complex-q parameters). Its best state is
expanded across the non-parametric rate nodes and initializes 64 independent
tile fits using deterministic least-squares MAP optimization for up to 10,000
evaluations. The fits use absolute semi-infinite HgCdTe power with free
positive n/k nuisance scales, independent tile AR parameters, lognormal initial
thickness priors (median 10 nm, log sigma 1.5), the 5 nm/day trajectory prior,
tile-rate and trajectory variation, and no neighboring-tile rate smoothing.
Corrections are expected to be already included in the input statistics; no
fixed-correction CSV is passed to the model. Existing tile archives are reused
only when their sufficient-statistics fingerprint matches the current input.

All production outputs are written below the input directory, not the
repository root:

```text
<statistics-dir>/production/<detector>/
  emcee/                    # one constant-rate initializer
  tile_statistics/          # one split CSV per tile
  fits/tile_000/            # one deterministic MAP archive per tile
  tile_rate_posterior.csv   # detector-level aggregation
```

`fit_wfi_tmm_production.py` contains only the fixed production TMM model and
MAP writer. The independent runner is the supported entrypoint for the current
cold-EMCEE plus per-tile least-squares production configuration.

Generate matching detector and all-detector plots from the same statistics
directory with:

```bash
.venv/bin/python -u plot_wfi_independent_production.py --root /path/to/corrected_statistics
.venv/bin/python -u plot_wfi_all_detector_trajectories.py \
  --root /path/to/corrected_statistics \
  --reset-day 15.064405023148149
.venv/bin/python -u plot_wfi_fitted_incidence_angle_mosaic.py \
  --root /path/to/corrected_statistics
```

The incidence-angle command writes
`production/wfi_fitted_incidence_angle_mosaic.png` below the statistics directory.

The original `fit_wfi_tmm_numpyro.py` remains the exploratory/reproduction
implementation and is not the production entrypoint.

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
.venv/bin/python3 fit_wfi_tmm_numpyro.py \
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

Generate matching plots with the same raw statistics:

```bash
.venv/bin/python3 -u plot_wfi_tmm_numpyro.py \
  --output-dir wfi01_absolute_substrate_power_grey_corrected_complex_ice_map5000_all_tiles \
  --tile-statistics rrn_reduction_20260930/wfi01_tile_log_statistics_512.csv
```

All reset-validation outputs and logs stay inside the repository workspace.

### Legacy NumPyro comparison workflow

The following options apply only to `fit_wfi_tmm_numpyro.py` comparison runs.
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
python fit_wfi_tmm_numpyro.py \
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

```bash
python plot_wfi_tmm_numpyro.py \
  --output-dir wfi_all_sca_tmm_numpyro_tile_production \
  --tile-statistics wfi_tile_log_statistics_512
```

The plotter writes the shared trajectory, posterior corners, per-SCA AR maps,
tile fit diagnostics, and observed-versus-modeled residual plots.

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

- `build_wfi01_tile_statistics.py`: tile sufficient-statistic calculation;
- `run_wfi_reduction_pipeline.py`: Nexus S3/MAST streaming, dated reduction,
  and count-space correction application;
- `build_wfi_tile_statistics_stream.py`: resumable MAST downloader;
- `run_wfi_production.py`: fixed all-detector independent-tile production runner;
- `run_wfi_tmm_emcee_warmup.py`: fixed one-tile EMCEE initializer;
- `fit_wfi_tmm_production.py`: fixed production TMM model and MAP writer;
- `fit_wfi_tmm_numpyro.py`: single- and all-SCA TMM model and output writer;
- `plot_wfi_tmm_numpyro.py`: posterior prediction and plotting;
- `analyze_rcs_soc_ice_model.py`: shared trajectory and sufficient-statistic utilities;
- `fit_wfi01_joint_complex_q_soc.py`: response integration and effective-response diagnostics;
- `fit_wfi01_effective_boundary_numpyro.py`: separate effective-boundary diagnostic;
- `test_*.py`: synthetic and local fixture tests.

## Verification

Run the tests from the project root:

```bash
python -m unittest test_fit_wfi_tmm_production.py test_run_wfi_production.py
python -m py_compile fit_wfi_tmm_production.py run_wfi_tmm_emcee_warmup.py run_wfi_production.py
python -m pytest -q
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
