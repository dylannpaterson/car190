# Roman WFI Ice Analysis

This repository contains the Roman WFI ice-monitoring analysis scripts. The
primary workflow converts calibrated MAST products into compact tile
sufficient statistics, then fits an air/ice/AR/HgCdTe transfer-matrix model
to the statistics with NumPyro.

The current production model is designed for all 18 WFI SCAs, fitted as
fresh-EMCEE-initialized independent per-tile MAP-only runs:

- absolute-substrate-power response with semi-infinite HgCdTe;
- positive free HgCdTe n/k nuisance scales;
- fully non-parametric ice trajectories with independent tile-level rate variation;
- independent AR parameters for each tile;
- lognormal initial-thickness priors centered at 10 nm;
- selectable one-, two-, or three-layer dielectric AR stack;
- no neighboring-tile rate smoothing.

The opt-in `--use-hgcdte-optical-constants` mode replaces the constant HgCdTe
boundary with a fixed 80 K, `x=0.445` wavelength-dependent sensitivity model.
It interpolates the 80 K real-index table in Mouzali (tel-01281981) between
the `x=0.397` and `x=0.459` columns and uses that source's absorption laws for
the imaginary index. The real-index table is held at its 1.6 um edge across
the shorter WFI wavelengths, so this mode is a documented sensitivity test,
not a detector-specific calibration. It is mutually exclusive with
`--fit-hgcdte`. Mouzali models a finite active HgCdTe layer; this workflow
reuses the constants only as an effective terminal boundary. The `x=0.445`
low-temperature extrapolation is outside the stated composition range of the
source's gap formula.

The opt-in `--fit-hgcdte-modulation` mode fits seven shared smooth log-space
multiplier nodes for each of `n(lambda)` and `k(lambda)` around that
wavelength-dependent baseline. Exponentiation keeps both optical constants
positive. This is a spectral-flexibility diagnostic shared across tiles, not
an independent detector calibration.

The opt-in `--fit-conductive-layer` mode inserts a global thin Drude-like layer
between the AR stack and HgCdTe. Its positive log-thickness, plasma-strength,
and damping-ratio parameters provide a low-dimensional physical boundary
perturbation without replacing the transfer-matrix model.

The opt-in `--fit-qe-rbf` mode applies normalized Gaussian RBF corrections to
the ground-measured QE weighting. Its default four nodes are at 650, 950,
1250, and 1600 nm; `--qe-rbf-node-wavelengths-nm`, `--qe-rbf-width-nm`, and
`--qe-rbf-prior-sigma-log` provide basis sensitivity controls. Use it with
`--np-modulation` to fit QE shape and the ice-rate trajectory jointly.

The opt-in `--use-asw-optical-constants` mode replaces the constant lossless
ice index with the wavelength-dependent 100 K amorphous-H2O n/k table from the
Mastrapa laboratory data set. The table is used as a physical sensitivity test,
not as a claim that the flight ice phase and temperature are known exactly.
Use `--ice-optical-constants-source` to point this sensitivity test at a local
copy instead of the pinned public URL.

The opt-in `--fit-ice-optical-modulation` mode fits six shared smooth log-space
multiplier nodes, placed at the six RCS SED maxima, for each of the Mastrapa
baseline `n(lambda)` and `k(lambda)` spectra. The default node prior sigma is
0.05. `--ice-n-modulation-prior-sigma-log` can broaden only the refractive-index
prior for porosity sensitivity tests. This mode is global across tiles and epochs
and is mutually exclusive with the scalar pre/post-reset ice-index modes; the
reset is represented by the shared thickness trajectory.

With `--fit-post-reset-tile-log-offset`, the fixed correction remains the
baseline while one zero-centered additive log correction is fitted per tile for
groups on or after `--ice-reset-day`. Positive values use the same convention as
the correction CSV and add to the observed corrected log ratios. The default
prior sigma is 0.1 and posterior values are written to
`post_reset_tile_log_offset_posterior.csv`.

The opt-in `--incidence-pupil-half-angle-deg` mode evaluates oblique s and p
responses over a two-dimensional circular pupil with uniform pupil-area
weighting. Roman's documented f/8 beam corresponds to a 3.576 degree
marginal-ray half-angle. Global chief-ray x/y tilts or a per-SCA/tile
`--incidence-chief-ray-map` can represent field-dependent geometry. The older
`--incidence-cone-half-angle-deg` solid-angle sensitivity remains available for
archive compatibility. Zero pupil size and chief-ray tilt retain the exact
normal-incidence calculation. `--fit-incidence-angle` instead fits one global
non-negative chief-ray polar angle relative to normal with a truncated-Normal
prior (default sigma 3 degrees) and a hard 15 degree upper bound. Its positive
initialization avoids the zero-angle symmetry point; it cannot be combined with
fixed chief-ray geometry or the legacy solid-angle cone. Use
`--incidence-angle-prior-uniform` for a flat 0--15 degree sensitivity run.

The production response uses `--sed-weighting energy`, treating the table values
as photon-rate weights. The `photon` interpretation remains available as a
sensitivity mode.

The production command fixes detector/tile/group gain and ACM corrections from
`refit_corrections_tile/wfi_all_tile_fixed_led3_gain_acm_corrections.csv`.
The table contains 18 detectors x 64 tiles x 215 groups; the loader selects the
matching detector and tile rows. ACM and early-time gain terms are therefore
not refit.

The production `--fully-np-trajectory` mode replaces the square-root accumulation law
with a positive instantaneous rate at every observation-epoch node. Rates are log-linearly
interpolated and integrated to preserve monotonic ice thickness; the existing
time-aware log-rate curvature penalty remains active. The node-rate prior is
centered on the historical initial-rate scale, but no square-root time dependence
is imposed. It is mutually exclusive with `--np-modulation`.

The older WFI01-only analyses and effective-boundary diagnostics remain in the
repository as comparison tools. They are not the default production model.

## Environment

The pinned base dependencies are in `requirements-wfi01-ice.txt`. The
NumPyro fit additionally requires working `jax`, `numpyro`, and `corner`
installations. The current development environment is named `bulge_env`.
`roman-mast-tools` 0.9.1 is not published on PyPI, so it is intentionally not
in the requirements file; install it separately from the supplied local source
archive after extracting it.

The production TMM computes its own transfer-matrix response and generates a
480--2700 nm wavelength grid from the public QE support. The historical
`--model-path` option is retained for command compatibility but is not read by
the production TMM; older SOC/effective-boundary tools still use that table.

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

The default filename pattern is
`{detector_lower}_tile_log_statistics_512_count_corrected.csv`. Use
`--statistics-pattern` for a different corrected-file naming convention. All
WFI01-WFI18 detectors are processed by default; repeat `--detector WFI##` to
select a subset. Use `--output-dir` to place detector outputs in another
repository directory, and `--dry-run` to validate all input files and paths
without fitting.

Each detector starts with a fresh 52-walker, 1,000-step EMCEE basin search
(seed `20261004`, no complex-q parameters). Its result initializes 64
independent tile fits using deterministic least-squares MAP optimization for up
to 10,000 steps. The fits use absolute semi-infinite HgCdTe power with free
positive n/k nuisance scales, independent tile AR parameters, lognormal initial
thickness priors (median 10 nm, log sigma 1.5), the 5 nm/day trajectory prior,
tile-rate and trajectory variation, and no neighboring-tile rate smoothing.
Corrections are expected to be already included in the input statistics; no
fixed-correction CSV is passed to the model. Existing tile archives are reused
only when their sufficient-statistics fingerprint matches the current input.

`fit_wfi_tmm_production.py` remains the underlying production TMM model and
configuration library. The independent runner is the supported entrypoint for
the current cold-EMCEE plus per-tile least-squares production configuration.

Generate matching detector and all-detector plots from a non-default output
directory with:

```bash
.venv/bin/python -u plot_wfi_independent_production.py --root /path/to/outputs
.venv/bin/python -u plot_wfi_all_detector_trajectories.py \
  --root /path/to/outputs \
  --fit-suffix absolute_substrate_power_hgcdte_prior025_independent_tiles_least_squares \
  --reset-day 15.064405023148149
```

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

Use `--ar-layers 1` or `--ar-layers 2` for comparison models. If the default
output path is used, they are written to paths ending in `_ar1layer` and
`_ar2layer` rather than overwriting the three-layer fit. Layer order is from
air toward HgCdTe.

For the reduced all-tile model, add `--shared-ar-optical-properties`. This
shares each layer's refractive index across every tile while retaining
hierarchical tile-to-tile thickness variation. The dielectric model fixes
`k=0` by default; `--fit-ar-absorption` is an explicit sensitivity option.

The general CLI settings include a three-layer dielectric AR stack, NUTS
sampling, all available tiles, exact SED wavelength nodes and stride 1,
log-noise sigma 0.005, and AR population-mean prior sigma 2.0. The current
production command above intentionally overrides this with deterministic
MAP-only fitting. The likelihood pixel weight defaults to the inverse median
valid-pixel count, converting each tile/group pixel SSE to approximately one
effective datum before the selected-tile average. The aggregate SSE is then
normalized by the number of selected tiles so adding tiles does not make NUTS
geometrically intractable. Use `--likelihood-weight` only for an explicit
sensitivity run.

Use `--fixed-corrections PATH` to apply fixed gain and first/late ACM
corrections to the log-ratio statistics. The production path is the
detector/tile-aware table above; this mode disables fitting the ACM and
early-time gain terms, preventing the same corrections from being fitted twice.

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
the production command uses both options. Neighboring selected tiles receive a
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

The fit directory contains:

- `posterior_samples.npz`: NumPyro posterior arrays, tile IDs, SCA names, and
  the selected AR layer count, archived response/statistics settings, and a
  sufficient-statistics fingerprint;
- `posterior_samples.npz` also records the public QE column selected for each SCA;
- `tile_ar_posterior.csv`: 16th/50th/84th percentile AR parameters per SCA/tile;
- `sca_ar_posterior.csv`: SCA-level AR population summaries for all-SCA fits;
- `np_node_posterior.csv`: epoch-node rate-multiplier and instantaneous-rate summaries when NP modulation is enabled;
- `tile_rate_posterior.csv`: per-SCA/tile rate offsets and instantaneous-rate summaries when tile-rate variation is enabled;
- `numpyro_summary.md`: settings and posterior percentile summary;
- PNG diagnostics generated by `plot_wfi_tmm_numpyro.py`.

The plotter uses the archived statistics path, SED mode, wavelength settings,
tile selection, and observation fingerprint by default. Pass an explicit
override only when intentionally plotting a compatible posterior subset.

## Code Map

- `build_wfi01_tile_statistics.py`: tile sufficient-statistic calculation;
- `run_wfi_reduction_pipeline.py`: Nexus S3/MAST streaming, dated reduction,
  and count-space correction application;
- `build_wfi_tile_statistics_stream.py`: resumable MAST downloader;
- `fit_wfi_tmm_numpyro.py`: single- and all-SCA TMM model and output writer;
- `plot_wfi_tmm_numpyro.py`: posterior prediction and plotting;
- `analyze_rcs_soc_ice_model.py`: shared trajectory and sufficient-statistic utilities;
- `fit_wfi01_joint_complex_q_soc.py`: response integration and effective-response diagnostics;
- `fit_wfi01_effective_boundary_numpyro.py`: separate effective-boundary diagnostic;
- `test_*.py`: synthetic and local fixture tests.

## Verification

Run the tests from the project root:

```bash
python -m pytest -q
python -m py_compile fit_wfi_tmm_numpyro.py plot_wfi_tmm_numpyro.py
```

The tests do not require raw MAST products. The MAST streaming tests use
injected image fetchers and the tracked index/statistics fixtures.

## Caveats

- The all-SCA implementation selects `SCU 1` through `SCU 18` from the public
  QE table for `WFI01` through `WFI18`, respectively. The official SCU-to-SCA
  serial mapping identifies WFI01/SCU 1 with SCA serial 22081.
- The production model generates its wavelength grid from public QE support;
  the external SCA1 response table is not used. It uses nominal fixed HgCdTe
  constants for every SCA by default because detector-specific constants are
  not currently available. The wavelength-dependent HgCdTe mode is an
  extrapolated 80 K composition sensitivity model, applied as an effective
  terminal boundary, and should not be treated as measured detector optical
  constants.
- The one-, two-, and three-layer models use the same broad per-layer AR
  bounds and hierarchical priors. They are effective dielectric-stack models,
  not claims about the proprietary physical coating recipe.
- The conductive-interlayer mode is a sensitivity model. It uses a fixed
  `epsilon_inf=1` and a reference wavelength of 1600 nm; the fitted Drude
  parameters are not detector-material measurements without external priors.
- The QE RBF mode is likewise a calibration sensitivity model. Its correction
  coefficients are global and prior-regularized; grey QE changes cancel under
  the per-LED reference normalization.
- The exact external effective-boundary equation is not available, so the
  effective-boundary scripts remain sensitivity diagnostics.
- Short smoke chains are useful for exercising code paths but are not
  scientifically interpretable posterior fits.
- Generated outputs and raw products are ignored by Git; retain important
  results separately if they need to be archived.
